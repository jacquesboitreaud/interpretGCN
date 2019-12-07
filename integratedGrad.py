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
        self.features_dim, self.n_rels = model.features_dim, model.num_rels
        self.model.train()
        
    def node_attrib(self, x, nodes_idx):
        # Attribution for a node or list of nodes
        if(nodes_idx==-1):
            nodes_idx=[i for i in range(len(x.nodes()))]
        ig = torch.zeros(len(nodes_idx),self.features_dim)
        # number of rectangles for integral approx : 
        m=50
        
        # Loop to compute integrated grad 
        x_h = x.ndata['h'].detach().numpy()
        
        for k in range(m):
             alpha=k/m
             with torch.no_grad():
                 x.ndata['h']=deepcopy(x_h) # reset features to initial
                 for i in nodes_idx:
                     # interpolate : 0 + (k/m) * embedding
                     x.ndata['h'][i]=0 + alpha* torch.tensor(x_h[i])
                     #print(x.ndata['h'])
                     
            #Compute grads 
             self.model.zero_grad()
             x.ndata['in']=x.ndata['h']
             x.ndata['in'].requires_grad_(True)
             out = self.model(x)
             if(k==0):
                 baseline_out = out.item()
                 print(baseline_out)
             elif(k==m-1):
                 x_out = out.item()
                 print(x_out)
             g=torch.autograd.grad(out, x.ndata['in'])[0] # list of gradients wrt inputs
             ig+=g[nodes_idx,]
             
             #print(g[lookup_indexes,])
        ig=ig/m
        return ig, x_out-baseline_out
    

        