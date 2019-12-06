# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 07:35:43 2019

@author: jacqu

Integrated gradients attribution to prediction for HERG binding 
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

from rgcn import Model
from molDataset import Loader, molDataset
from integratedGrad import IntegratedGradients


N_mols=13
# List all substructures
with open('data/vocab.txt','r') as f:
    vocab = f.readlines()
    vocab = [s.rstrip('\n') for s in vocab]
    
chem_att = {v: [0,0] for v in vocab} # dict of tuples (attention received, count occurences in set)


# Get vocabulary of substructures 
vocab_dict = {s:i for (i,s) in enumerate(vocab)}

loader = Loader(csv_path='data/moses_train.csv',
                 n_mols=N_mols,
                 num_workers=0, 
                 batch_size=4, 
                 shuffled= True,
                 target = 'logP',
                 test_only=True)
rem, ram, rchim, rcham = loader.get_reverse_maps()
_ ,_ , test_loader = loader.get_data()

# # Instantiate IG + load model 
model_path= 'saved_model_w/logp.pth'
params = pickle.load(open('saved_model_w/params.pickle','rb'))
params['classifier']=False


model = Model(**params)
model.load_state_dict(torch.load(model_path))
inteGrad = IntegratedGradients(model)


# Get first molecule of first batch 
graph, target = next(iter(test_loader))
graphs = dgl.unbatch(graph)
x, target = graphs[0], target[0]
out = model(x)
print(f'Predicted logp is {out.item()}, true is {target.item()}')
attrib = inteGrad.attribute(x, -1)

# Problem : each embedding is 16-dimensional at the moment ... 
x.edata['ig']=attrib[:,5] 

# Select + and - edges (atoms):
x=x.to_networkx(node_attrs=['atomic_num','chiral_tag','formal_charge','num_explicit_hs','is_aromatic'], 
                    edge_attrs=['one_hot','ig'])
x=x.to_undirected()
positives, negatives =set(), set()
for (src,dst, data) in x.edges(data=True):
    if(data['ig'].item()>0):
        positives.add(src)
        positives.add(dst)
    elif(data['ig'].item()<0):
        negatives.add(src)
        negatives.add(dst)

# To networkx and plot with colored bonds 
img=highlight(mol,list(positives), color= [0,1,0])
img2=highlight(mol,list(negatives), color=[1,0,0])