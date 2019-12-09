# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:52:46 2019

@author: jacqu

Distribution of attention over substructures, over the test set 

Run in working dir ../
"""
import torch
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt

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
    
    N_mols=100
    # List all substructures
    with open('data/vocab.txt','r') as f:
        vocab = f.readlines()
        vocab = [s.rstrip('\n') for s in vocab]
        
    chem_att = {v: [0,0] for v in vocab} # dict of tuples (attention received, count occurences in set)
    
    
    # Get vocabulary of substructures 
    vocab_dict = {s:i for (i,s) in enumerate(vocab)}
    
    loader = Loader(csv_path='data/test_set.csv',
                     n_mols=N_mols,
                     num_workers=0, 
                     batch_size=2, 
                     shuffled= True,
                     target = 'logP',
                     test_only=True)
    rem, ram, rchim, rcham = loader.get_reverse_maps()
    _ ,_ , test_loader = loader.get_data()
    
    # Load model 
    model_path= 'saved_model_w/logp.pth'
    params = pickle.load(open('saved_model_w/params.pickle','rb'))
    model = Model(**params)
    model.load_state_dict(torch.load(model_path))
    
    
    # Test set pass, keep attention weights 
    model.eval()
    t_loss=0
    for batch_idx, (graph,target) in enumerate(test_loader): 
            with torch.no_grad():
                
                target=target.view(-1,1) # Graph-level target : (batch_size,)
                #graph=send_graph_to_device(graph,model.device)
                out=model(graph).view(-1,1)
                
                t_loss+=F.mse_loss(out,target,reduction='sum')
                    
                # Transform graph to RDKit molecule for nice visualization
                graphs = dgl.unbatch(graph)
                
                for i in range(len(graphs)): # for each molecule in batch
                    g0=graphs[i]
                    n_nodes = len(g0.nodes)
                    att= get_attention_map(g0, src_nodes=g0.nodes(), dst_nodes=g0.nodes(), h=1)
                    att_g0 = att[0] # get attn weights only for g0
                    #Column i contains attention weights for edges coming into node i. 
                    
                    # Find atoms that receive extra attention from their neighbors
                    atoms = []
                    for dest in range(n_nodes):
                        diff = att_g0[:,dest] - 1/g0.in_degree(dest)
                        atoms += list(np.where(diff>0)[0])
                    atoms=np.unique(atoms)
                        
                    try:
                        mol = nx_to_mol(g0, rem, ram, rchim, rcham)
                        #img=highlight(mol,list(atoms))
                
                        for v in vocab: 
                            # Get atoms that match substructure 
                            matches = set(mol.GetSubstructMatch(Chem.MolFromSmiles(v)))
                            if(len(matches)>0):
                                chem_att[v][1]+=1 # substructure occurence
                                chem_att[v][0]+=len(matches.intersection(set(atoms))) # substructure attention (at least 1 atom in substructure)
                    except:
                        continue
    
    #Compute frequencies for substructures in the dataset 
    # Divide by number of atoms in each substructure
    
    freqs = {k: v[0]/(v[1]*num_atoms(k)) for (k,v) in chem_att.items() if v[1]>10}
    avg_loss = t_loss / N_mols
    
    # Non zero freqs : 
    std = sorted(freqs.items(), key=lambda x: x[1])
    goods = [kv for kv in std if 'O' in kv[0] or 'N' in kv[0] or 'F' in kv[0] or 'Cl' in kv[0] or 'Br' in kv[0]]
    
    sns.barplot(x=np.arange(len(goods)), y=[kv[1] for kv in goods])

    fig=plt.figure(dpi=300, figsize=(20,20))
    img=draw_multi([g[0] for g in goods])