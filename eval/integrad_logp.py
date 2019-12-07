# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 07:35:43 2019

@author: jacqu

Integrated gradients attribution to prediction for the logP prediction model, with 1-dimensional edge embeddings 
"""

import torch
import numpy as np 
import pandas as pd 
import pickle
import dgl


import sys
sys.path.append('./dataloading')
from utils import *
from draw_mol import *
from rdkit_to_nx import *
from viz import *

from rgcn_onehot import Model
from molDataset import Loader, molDataset
from integratedGrad import IntegratedGradients


N_mols=2
# List all substructures
with open('data/vocab.txt','r') as f:
    vocab = f.readlines()
    vocab = [s.rstrip('\n') for s in vocab]
chem_att = {v: [0,0] for v in vocab} # dict of tuples (attention received, count occurences in set)
# Get vocabulary of substructures 
vocab_dict = {s:i for (i,s) in enumerate(vocab)}


loader = Loader(csv_path='data/handmade.csv',
                 n_mols=N_mols,
                 num_workers=0, 
                 batch_size=2, 
                 shuffled= True,
                 target = 'logP',
                 test_only=True)
rem, ram, rchim, rcham = loader.get_reverse_maps()
_ ,_ , test_loader = loader.get_data()

# # Instantiate IG + load model 
model_path= 'saved_model_w/logp_lowd.pth'
params = pickle.load(open('saved_model_w/params.pickle','rb'))
params['classifier']=False


model = Model(**params)
model.load_state_dict(torch.load(model_path))
inteGrad = IntegratedGradients(model)


# Get first molecule of first batch 
m = 1
nodes = -1
feat=8 # which feature (one hots )


graph, target = next(iter(test_loader))
graphs = dgl.unbatch(graph)
x, target = graphs[m], target[m]
out = model(graph)[m]
print(f'Predicted logp is {out.item()}, true is {target.item()}')
attrib, delta = inteGrad.node_attrib(x, nodes)

# Problem : each embedding is 16-dimensional at the moment ... 
x.ndata['ig']=attrib

print(torch.sum(attrib).item(), delta)

# Select + and - edges (atoms):
x=x.to_networkx(node_attrs=['atomic_num','chiral_tag','formal_charge','num_explicit_hs','is_aromatic','ig'], 
                    edge_attrs=['one_hot'])
x=x.to_undirected()
positives, negatives =set(), set()
for (i, data) in x.nodes(data=True):
    if(data['ig'][feat].item()>0):
        positives.add(i)
    elif(data['ig'][feat].item()<0):
        negatives.add(i)


#TODO :adapt nx to mol function so that it can handle one-hot vectors for features !!!!!!!!!!!!!!!
mol=nx_to_mol(x,rem, ram, rchim, rcham )
# To networkx and plot with colored bonds 
img=highlight(mol,list(positives), color= [0,1,0]) #green
img2=highlight(mol,list(negatives), color=[1,0,0]) #red