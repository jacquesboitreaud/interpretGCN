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


N_mols=1000

pred, true = [],[]


loader = Loader(csv_path='data/ESR1_test.csv',
                 n_mols=N_mols,
                 num_workers=0, 
                 batch_size=100, 
                 shuffled= True,
                 target = 'binary',
                 test_only=True,
                 dude=True)
rem, ram, rchim, rcham = loader.get_reverse_maps()
_ ,_ , test_loader = loader.get_data()

# # Instantiate IG + load model 
model_path= 'saved_model_w/esr1.pth'
params = pickle.load(open('saved_model_w/params.pickle','rb'))
params['classifier']=True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = Model(**params).to(device)
model.load_state_dict(torch.load(model_path))
inteGrad = IntegratedGradients(model)

for i, (graph,target) in enumerate(test_loader):
    target=target.view(-1,1) # Graph-level target : (batch_size,)
    # Embedding for each node
    graph=send_graph_to_device(graph,device)
    out = model(graph).view(-1,1).cpu().detach().numpy()
    out, target = [o[0] for o in out], [t[0] for t in target.cpu().numpy()]
    
    pred += out
    true += target
    
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(true, pred)
    
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()