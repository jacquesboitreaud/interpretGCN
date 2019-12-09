# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 07:35:43 2019

@author: jacqu

Integrated gradients attribution to prediction for HERG binding prediction model
Run in ../
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


N_mols=3


loader = Loader(csv_path='data/esr1.csv',
                 n_mols=N_mols,
                 num_workers=0, 
                 batch_size=3, 
                 shuffled= False,
                 target = 'binary',
                 test_only=True,
                 dude=True)
rem, ram, rchim, rcham = loader.get_reverse_maps()
_ ,_ , test_loader = loader.get_data()

# # Instantiate IG + load model 
model_path= 'saved_model_w/esr1.pth'
params = pickle.load(open('saved_model_w/params.pickle','rb'))
params['classifier']=True


model = Model(**params)
model.load_state_dict(torch.load(model_path))
inteGrad = IntegratedGradients(model)

# ============================================================================
# Get first molecule of first batch 
m = 2
nodes = -1


graph, target = next(iter(test_loader))
graphs = dgl.unbatch(graph)
x, target = graphs[m], target[m]
out = model(graph)[m]
print(f'Predicted binding proba is {out.item()}, true is {target.item()}')
attrib, _ , delta = inteGrad.attrib(x, nodes)

# Attrib to dataframe 
df = pd.DataFrame(attrib.numpy())
df.columns = ['charge 0', 'charge +1', 'charge -1', 'H','Br','C','N','O','F','P','S','Cl','I']


print(torch.sum(attrib).item(), delta)
sns.heatmap(df.transpose(), vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
plt.xlabel('Node n°')




# Select + and - edges (atoms):
x.ndata['ig']=attrib # add attributions as a node feature
x=x.to_networkx(node_attrs=['atomic_num','chiral_tag','formal_charge','num_explicit_hs','is_aromatic','ig'], 
                    edge_attrs=['one_hot'])
x=x.to_undirected()
node_contribs = {'atom type':[], 'charge':[]}
for (i, data) in x.nodes(data=True):
    at_charge, at_type = torch.argmax(data['formal_charge']).item(), torch.argmax(data['atomic_num']).item()
    node_contribs['charge'].append(data['ig'][at_charge].item())
    node_contribs['atom type'].append(data['ig'][len(rcham)+at_type].item())
        
# Highlighting : Relative to other contributions 
"""
mean_t, mean_c = np.mean(node_contribs['atom type']), np.mean(node_contribs['charge'])
sd_t, sd_c = np.std(node_contribs['atom type']), np.std(node_contribs['charge'])

z_t= (node_contribs['atom type']-mean_t)/sd_t
z_c= (node_contribs['charge']-mean_c)/sd_c
"""
z_t=np.array(node_contribs['atom type'])
z_c=np.array(node_contribs['charge'])

pos = [int(i) for i in list(np.where(z_c+z_t>0)[0])]
neg = [int(i) for i in list(np.where(z_c+z_t<0)[0])]

# Plot heatmap of node contributions: 
plt.figure()
#df2=pd.DataFrame.from_dict({'atom type':z_t, 'charge':z_c, 'sum':z_t+z_c})
df2=pd.DataFrame.from_dict({'Attribution':z_t+z_c})
sns.barplot(x=np.arange(df2.shape[0]),y=list(2*df2['Attribution']), color='lightBlue')
plt.xlabel('Atom n°')
plt.ylabel('Attribution')

# For molecule plot with highlights, 
mol=nx_to_mol(x,rem, ram, rchim, rcham )
# To networkx and plot with colored bonds 
labels=['+','-']
img =highlight(mol,(tuple(pos),tuple(neg)),labels)