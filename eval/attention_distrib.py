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
        vocab = [s.rstrip('\n') for s in vocab]
        
    chem_att = {v: [0,0] for v in vocab} # dict of tuples (attention received, count occurences in set)
    
    
    # Get vocabulary of substructures 
    vocab_dict = {s:i for (i,s) in enumerate(vocab)}
    
    loader = Loader(csv_path='data/test_set.csv',
                     n_mols=2,
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
    for batch_idx, (graph,target) in enumerate(test_loader): 
            with torch.no_grad():
                
                target=target.view(-1,1) # Graph-level target : (batch_size,)
                #graph=send_graph_to_device(graph,model.device)
                out=model(graph).view(-1,1)
                
                t_loss=F.mse_loss(out,target,reduction='sum')
                    
                # Transform graph to RDKit molecule for nice visualization
                graphs = dgl.unbatch(graph)
                g0=graphs[0]
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
                    
                mol = nx_to_mol(g0, rem, ram, rchim, rcham)
                img=highlight(mol,list(atoms))
                
                for v in vocab: 
                    # Get atoms that match substructure 
                    matches = set(mol.GetSubstructMatch(Chem.MolFromSmiles(v)))
                    if(len(matches)>0):
                        chem_att[v][1]+=1 # substructure occurence
                    if(len(matches.intersection(set(atoms)))>0):
                        chem_att[v][0]+=1 # substructure attention (at least 1 atom in substructure)
    
    #Compute frequencies for substructures in the dataset 
    freqs = {k: v[0]/v[1] for (k,v) in chem_att.items() if v[1]>0}
    
    # Analyse 
                        