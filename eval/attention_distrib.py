# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:52:46 2019

@author: jacqu

Distribution of attention over substructures, over the test set 
"""
import torch
import torch.nn.functional as F

import pandas as pd 
import numpy as np 
import pickle
from rdkit import Chem

import sys
sys.path.append('dataloading')
from rgcn import Model
from molDataset import Loader

from utils import *
from draw_mol import *
from rdkit_to_nx import *
from viz import *


if(__name__=='__main__'):
    # List all substructures
    with open('data/vocab.txt','r') as f:
        vocab = f.readlines()
    
    
    # Get vocabulary of substructures 
    vocab_dict = {s.rstrip('\n'):i for (i,s) in enumerate(vocab)}
    molecules = pd.read_csv('data/test_set.csv', nrows=10)
    smiles = molecules['can']
    
    loader = Loader(csv_path='data/test_set.csv',
                     n_mols=14,
                     num_workers=0, 
                     batch_size=10, 
                     shuffled= True,
                     target = 'logP',
                     test_only=True)
    _ ,_ , test_loader = loader.get_data()
    
    # Load model 
    model_path= 'saved_model_w/logp.pth'
    params = pickle.load(open('saved_model_w/params.pickle','rb'))
    model = Model(**params)
    model.load_state_dict(torch.load(model_path))
    # Test set pass, keep attention weights 
    
    
    # Iterate on molecules in test set 
    model.eval()
    for batch_idx, (graph,target) in enumerate(test_loader): 
            with torch.no_grad():
                
                target=target.view(-1,1) # Graph-level target : (batch_size,)
                #graph=send_graph_to_device(graph,model.device)
                out=model(graph).view(-1,1)
                
                t_loss=F.mse_loss(out,target,reduction='sum')
                    
                # Transform graph to RDKit molecule for nice visualization
                graphs = dgl.unbatch(graph)
                g0=graphs[1]
                n_nodes = len(g0.nodes)
                att= get_attention_map(g0, src_nodes=g0.nodes(), dst_nodes=g0.nodes(), h=1)
                att_g0 = att[0] # get attn weights only for g0
                
                # Select atoms with highest attention weights and plot them 
                tops = np.unique(np.where(att_g0>0.55)) # get top atoms in attention
                mol = nx_to_mol(g0, rem, ram, rchim, rcham)
                img=highlight(mol,list(tops))
                
                for v in vocab: 
                    print(v)
                    # Retrieve attention weights and aggregate 
                    #mol1.GetSubstructMatch(mol2)