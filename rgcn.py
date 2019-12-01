# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:44:23 2019

@author: jacqu

RGCN to learn node embeddings on RNA graphs , with edge labels 

https://docs.dgl.ai/tutorials/models/1_gnn/4_rgcn.html#sphx-glr-tutorials-models-1-gnn-4-rgcn-py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from functools import partial
import dgl
from dgl import mean_nodes

from dgl.nn.pytorch.glob import SumPooling
from dgl.nn.pytorch.conv import GATConv

""" Graph attention layer to be able to visualize attention on edges """
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        # equation (1)
        z = self.fc(h)
        g.ndata['z'] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')
    
""" RGCN layer to propagate message on different edge types """
class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False, features_dim=None):
        super(RGCNLayer, self).__init__()
        self.in_dim=in_dim
        if(is_input_layer):
            self.n_atoms, self.n_charges = features_dim # to keep in memory for atom embedding at input layer 
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer

        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # weight bases in equation (3)
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_dim,
                                                self.out_dim))
        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_dim))

        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases (equation (3))
            weight = self.weight.view(self.in_dim, self.num_bases, self.out_dim)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                        self.in_dim, self.out_dim)
        else:
            weight = self.weight
            
        if self.is_input_layer:
            def message_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node atom type 
                embed = weight.view(-1, self.out_dim)
                index = edges.data['one_hot'] * self.n_atoms \
                + edges.src['atomic_num']*self.n_charges \
                +edges.src['formal_charge']
                return {'msg': embed[index]}
        else:
            def message_func(edges):
                w = weight[edges.data['one_hot']]
                # print(w.size(),w)
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                return {'msg': msg}


        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                h = h + self.bias
            if self.activation:
                h = self.activation(h)
            return {'h': h}


        g.set_n_initializer(dgl.init.zero_initializer)
        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)


class Model(nn.Module):
    # Computes embeddings for all nodes
    # No features
    def __init__(self, atom_types, charges, h_dim, out_dim , num_rels, num_bases=-1, num_hidden_layers=2):
        super(Model, self).__init__()
        
        self.atom_types, self.charges, self.h_dim, self.out_dim = atom_types, charges, h_dim, out_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_rels = num_rels
        self.num_bases = num_bases
        # create rgcn layers
        self.build_model()
        
        
        self.attn = GATLayer(in_dim=self.out_dim, out_dim=self.out_dim)
        self.dense = nn.Linear(self.out_dim,1)
        self.pool = SumPooling()

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer(out_dim=self.out_dim)
        self.layers.append(h2o)
        
    def build_input_layer(self):
        return RGCNLayer(self.atom_types*self.charges, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=True, features_dim=(self.atom_types,self.charges))

    def build_hidden_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu)

    def build_output_layer(self, out_dim):
        return RGCNLayer(self.h_dim, out_dim, self.num_rels, self.num_bases)


    def forward(self, g):
        #print('edge data size ', g.edata['one_hot'].size())
        for layer in self.layers:
             #g.ndata['h']=torch.cat([g.ndata['formal_charge'].unsqueeze(1), g.ndata['atomic_num'].unsqueeze(1)], dim=1)
             #print(g.ndata['h'].size())
             layer(g)
             #print(g.ndata['h'].size())
        attention = self.attn(g,g.ndata['h'].view(-1,self.out_dim))
        
        out=self.pool(g,attention)
        out=self.dense(out)
        #print(out.shape)
        
        # Return node embeddings 
        #return g.ndata['h']
        return out