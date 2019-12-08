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
        
    def attrib(self, x, nodes_idx):
        # Attribution for a node or list of nodes
        if(nodes_idx==-1):
            nodes_idx=[i for i in range(len(x.nodes()))]
        elif(type(nodes_idx)==int):
            nodes_idx=[nodes_idx]
        ig_nodes = torch.zeros(len(x.nodes()),self.features_dim)
        
        #ig_ed = torch.zeros(self.n_rels, self.features_dim)
        
        # number of rectangles for integral approx : 
        m=100
        
        # Loop to compute integrated grad 
        x_h = x.ndata['h'].detach().numpy()
        x_e = self.model.layers[0].weight.detach().numpy()
        
        for k in range(m):
             alpha=k/m
             with torch.no_grad():
                 x.ndata['h']=deepcopy(x_h) # reset features to initial
                 #self.model.layers[0].weight=deepcopy(torch.nn.Parameter(torch.tensor(x_e))) # reset features to initial
                 
                 # interpolate : 0 + (k/m) * embedding
                 x.ndata['h']=0 + alpha* torch.tensor(x_h)
                 self.model.layers[0].weight = torch.nn.Parameter(torch.tensor(0 + alpha*x_e))
                 #print(self.model.layers[0].weight)
                 #print(x.ndata['h'][i]) 
                     
            #Compute grads 
             self.model.zero_grad()
             x.ndata['in']=x.ndata['h']
             x.ndata['in'].requires_grad_(True)
             
             out = self.model(x)
             if(k==0):
                 baseline_out = out.item()
                 #print('Baseline output:',baseline_out)
             elif(k==m-1):
                 x_out = out.item()
                 #print('x output:', x_out)
             g=torch.autograd.grad(out, [x.ndata['in'], self.model.layers[0].weight]) # list of gradients wrt inputs
             ig_nodes += g[0]
             #ig_ed += torch.sum(g[1],dim=2)
             
             #print(g[lookup_indexes,])
        ig_nodes, ig_ed = (x.ndata['in']-0)* ig_nodes/m, 0
        return ig_nodes.detach(), ig_ed, x_out-baseline_out
    

        