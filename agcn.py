# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:44:23 2019

@author: jacqu

GCN to learn node embeddings on RNA graphs , with edge labels 
- 1 attention layer 
- num_hidden_layers + 2 rgcn layers

source : 
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
from dgl.nn.pytorch.conv import GATConv, RelGraphConv

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
        return {'e' : F.leaky_relu(a)}
    
    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z' : edges.src['z'], 'e' : edges.data['e']}
    
    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h' : h}
    
    def forward(self,g, h):
        # equation (1)
        z = self.fc(h)
        g.ndata['z'] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata['z']



class Model(nn.Module):
    # Computes embeddings for all nodes
    # No features
    def __init__(self, features_dim, h_dim, out_dim , num_rels, num_bases=-1, num_hidden_layers=2, classifier=False):
        super(Model, self).__init__()
        
        self.features_dim, self.h_dim, self.out_dim = features_dim, h_dim, out_dim
        
        self.attn = GATLayer(in_dim=self.features_dim, out_dim=self.h_dim)
        
        self.num_hidden_layers = num_hidden_layers
        self.num_rels = num_rels
        self.num_bases = num_bases
        # create rgcn layers
        self.build_model()
        
        self.dense = nn.Linear(self.out_dim,1)
        self.pool = SumPooling()
        self.is_classifier=classifier

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        #i2h = RelGraphConv(self.h_dim, self.h_dim, self.num_rels, activation=nn.ReLU())
        #self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = RelGraphConv(self.h_dim, self.h_dim, self.num_rels, activation=nn.ReLU())
            self.layers.append(h2h)
        # hidden to output
        h2o = RelGraphConv(self.h_dim, self.out_dim, self.num_rels, activation=nn.ReLU())
        self.layers.append(h2o)


    def forward(self, g):
        #print('edge data size ', g.edata['one_hot'].size())
        
        #GAT Layer
        g.ndata['h']=self.attn(g,g.ndata['h'])
        
        for layer in self.layers:
             #print(g.ndata['h'].size())
             #print(g.edata['one_hot'].size())
             g.ndata['h']=layer(g,g.ndata['h'],g.edata['one_hot'])
             
        
        out=self.pool(g,g.ndata['h'].view(len(g.nodes),-1,self.out_dim))
        out=self.dense(out)
        if(self.is_classifier):
            out=torch.sigmoid(out)
        #print(out.shape)
        
        return out