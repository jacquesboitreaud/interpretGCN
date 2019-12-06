# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 06:47:06 2019

@author: jacqu

Integrated gradients class to compute IG 
"""
import torch
import torch.autograd
import numpy as np
import dgl
from copy import deepcopy

class IntegratedGradients():
    def __init__(self, model):
        self.model = model 
        self.n_atoms, self.n_charges, self.n_rels = model.atom_types, model.charges, model.num_rels
        self.model.train()
        
    def attribute(self, x, edges_idx):
        # x is input graph (dgl), edges_idx is a list of the indices of edges for which we examine the contribution
        if(edges_idx==-1):
            edges_idx=np.arange(len(x.edges()[0])) # Do it for all edges in graph
        ig = torch.zeros(len(edges_idx),self.model.layers[0].out_dim)
        
        # number of rectangles for integral approx : 
        m=20
        
        # x is input, edge_index is the index of the edge we remove (in x.edges())
        edges=x.find_edges(edges_idx)
        src_nodes = edges[0] # we don't need destination nodes
        lookup_indexes = [x.edata['one_hot'][i] * self.n_atoms \
                + x.ndata['atomic_num'][src]*self.n_charges \
                +x.ndata['formal_charge'][src] for i,src in enumerate(src_nodes)]
         
        # Loop to compute integrated grad 
        embed = deepcopy(self.model.layers[0].weight.view(-1,self.model.layers[0].out_dim).detach().numpy())
        for k in range(m):
             alpha=k/m
             with torch.no_grad():
                 for i in lookup_indexes:
                     # interpolate : 0 + (k/m) * embedding
                     self.model.layers[0].weight.view(-1,self.model.layers[0].out_dim)[i]=alpha*torch.tensor(embed[i]) 
                     
                     #print interpolated embeddings
                     #print(self.model.layers[0].weight.view(-1,self.model.layers[0].out_dim)[i])
                     
             
             #Compute grads 
             out = self.model(x)
             torch.autograd.backward(out, self.model.layers[0].weight)
             g= self.model.layers[0].weight.grad.view(-1,self.model.layers[0].out_dim)
             #g = torch.autograd.grad(out[0],self.model.layers[0].weight, allow_unused=True)
             ig+=g[lookup_indexes,]
             #print(g[lookup_indexes,])
             #print(ig)
        
        return ig
    

        