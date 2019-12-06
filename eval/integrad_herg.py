# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 07:35:43 2019

@author: jacqu

Integrated gradients attribution to prediction for HERG binding 
"""

import torch
import numpy as np 
import pandas as pd 
from copy import deepcopy


import sys
sys.path.append('./dataloading')
from rgcn import Model
from molDataset import Loader
from integratedGrad import IntegratedGradients


N_mols=100
# List all substructures
with open('data/vocab.txt','r') as f:
    vocab = f.readlines()
    vocab = [s.rstrip('\n') for s in vocab]
    
chem_att = {v: [0,0] for v in vocab} # dict of tuples (attention received, count occurences in set)


# Get vocabulary of substructures 
vocab_dict = {s:i for (i,s) in enumerate(vocab)}

loader = Loader(csv_path='data/HERG_dataset.csv',
                 n_mols=N_mols,
                 num_workers=0, 
                 batch_size=10, 
                 shuffled= True,
                 target = 'binary',
                 test_only=True)
rem, ram, rchim, rcham = loader.get_reverse_maps()
_ ,_ , test_loader = loader.get_data()

# # Instantiate IG + load model 
model_path= 'saved_model_w/herg.pth'
params = pickle.load(open('saved_model_w/params.pickle','rb'))
model = Model(**params)
model.load_state_dict(torch.load(model_path))
inteGrad = IntegratedGradients(model)



# Get first batch 
batch = next(iter(test_loader))
graphs = dgl.unbatch(graph)

sample = graphs[0]

baseline = inteGrad.baseline(sample)

attrib = inteGrad.attribute(sample, baseline)

