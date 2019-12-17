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
from agcn import Model
from molDataset import Loader

from utils import *
from draw_mol import *
from rdkit_to_nx import *
from viz import *


if(__name__=='__main__'):
    
    N_mols=1000
    # List all substructures
    with open('data/vocab.txt','r') as f:
        vocab = f.readlines()
        vocab = [s.rstrip('\n') for s in vocab]
    
    # dict of tuples (attention received, count occurences in set)
    chem_att = {v: [0,0] for v in vocab}
    resi_df = {'true':[], 'pred':[]}
    
    
    # Get vocabulary of substructures 
    vocab_dict = {s:i for (i,s) in enumerate(vocab)}
    
    loader = Loader(csv_path='data/test_set.csv',
                     n_mols=N_mols,
                     num_workers=0, 
                     batch_size=100, 
                     shuffled= False,
                     target = 'logP',
                     test_only=True)
    rem, ram, rchim, rcham = loader.get_reverse_maps()
    _ ,_ , test_loader = loader.get_data()
    
    # Load model 
    model_path= 'saved_model_w/logp_attn.pth'
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
                    resi_df['true'].append(target.cpu()[i].item())
                    resi_df['pred'].append(out.cpu()[i].item())
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
    
    freqs = {k: v[0]/(v[1]*num_atoms(k)) for (k,v) in chem_att.items() if v[1]>0}
    avg_loss = t_loss / N_mols
    
    # Sort substructures by increasing coefficient 
    std = sorted(freqs.items(), key=lambda x: x[1])
    # Barplot of distribution 
    sns.barplot(x=np.arange(len(std)), y=[kv[1] for kv in std])
    # Drawing substructures 
    fig=plt.figure(dpi=300, figsize=(25,25))
    img=draw_multi([g[0] for g in std])
    
    #np.save('../results/sorted_freqs.npy',std)
    #np.save('../results/residuals.npy',resi_df)
    
    #TODO Random permutation test 
    
    """
    # Residuals plot: 
    sns.scatterplot(resi_df['true'], resi_df['pred'])
    plt.plot(np.arange(-3,7,1),np.arange(-3,7,1), color='r')
    plt.xlim(-3,6)
    plt.ylim(-3,6)
    """