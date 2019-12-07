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

import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = 8, 4


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


loader = Loader(csv_path='data/CHEMBL_18t.csv',
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

# ============================================================================
# Get first molecule of first batch 
m = 0
nodes = -1
feat= 8 # which feature (one hots )


graph, target = next(iter(test_loader))
graphs = dgl.unbatch(graph)
x, target = graphs[m], target[m]
out = model(graph)[m]
print(f'Predicted logp is {out.item()}, true is {target.item()}')
attrib, _ , delta = inteGrad.attrib(x, nodes)

# Attrib to dataframe 
df = pd.DataFrame(attrib.numpy())
df.columns = ['charge 0', 'charge +1', 'charge +2', 'charge -1', 'Br','B','C','N','O','F','P','S','Cl','I']


print(torch.sum(attrib).item(), delta)
sns.heatmap(df.transpose(), vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
plt.xlabel('Node n°')




# Select + and - edges (atoms):
x.ndata['ig']=attrib # add attributions as a node feature
x=x.to_networkx(node_attrs=['atomic_num','chiral_tag','formal_charge','num_explicit_hs','is_aromatic','ig'], 
                    edge_attrs=['one_hot'])
x=x.to_undirected()
positives, negatives =set(), set()
node_contribs = {'atom':[], 'charge':[]}
for (i, data) in x.nodes(data=True):
    at_charge, at_type = torch.argmax(data['formal_charge']).item(), torch.argmax(data['atomic_num']).item()
    node_contribs['charge'].append(data['ig'][at_charge].item())
    node_contribs['atom'].append(data['ig'][3+at_type].item())
    
    # Highlighting : Relative to other contributions 
    if(node_contribs['atom'][-1]>0):
        positives.add(i)
    elif(node_contribs['atom'][-1]<0):
        negatives.add(i)

# Plot heatmap of node contributions: 
plt.figure()
df2=pd.DataFrame.from_dict(node_contribs)
sns.heatmap(df2.transpose(), vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
plt.xlabel('Node n°')

#TODO :adapt nx to mol function so that it can handle one-hot vectors for features !!!!!!!!!!!!!!!
mol=nx_to_mol(x,rem, ram, rchim, rcham )
# To networkx and plot with colored bonds 
img,labels=highlight(mol,list(positives), color= [0,1,0]) #green
img2,_ =highlight(mol,list(negatives), color=[1,0,0]) #red