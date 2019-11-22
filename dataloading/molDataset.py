# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:06:44 2019

@author: jacqu

Dataset class for SMILES to graph 

"""

import os 
import sys
if __name__ == "__main__":
    sys.path.append("..")
    
import torch
import dgl
import pandas as pd
import numpy as np

import pickle

import networkx as nx

from torch.utils.data import Dataset, DataLoader, Subset

from rdkit_to_nx import smiles_to_nx


def collate_block(samples):
    # Collates samples into a batch
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    
    labels = torch.tensor(labels)
    
    return batched_graph, labels


class molDataset(Dataset):
    """ 
    pytorch Dataset for training on pairs of nodes of RNA graphs 
    """
    def __init__(self, csv_path="../../data/chembl_18t.csv",
                n_mols = 100,
                emb_size=1,
                debug=False, 
                shuffled=False):
        
        self.df = pd.read_csv(csv_path, nrows=n_mols)
        self.n = n_mols
        self.emb_size = emb_size
        
        # Choose targets for supervision: 
        #self.targets = np.load('../targets_chembl.npy') # load list of targets
        self.targets = ['logP']
        print(f'Labels retrieved for the following {len(self.targets)} targets: {self.targets}')
        
        # Load edge map
        with open('edges_and_nodes_map.pickle',"rb") as f:
            self.edge_map= pickle.load(f)
            self.at_map = pickle.load(f)
            self.chi_map= pickle.load(f)
            self.charges_map = pickle.load(f)
            
        self.num_edge_types, self.num_atom_types = len(self.edge_map), len(self.at_map)
        self.num_charges, self.num_chir = len(self.charges_map), len(self.chi_map)
        print('Loaded edge and atoms types maps.')
        
        if(debug):
            # special case for debugging
            pass
            
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        # gets the RNA graph n°idx in the list
        # Annotated pickle files are tuples (g, dict of rmsds between nodepairs)
        
        row = self.df.iloc[idx]
        print(row)
        
        smiles, targets = row['can'], np.array(row[self.targets],dtype=np.float32)
        
        graph=smiles_to_nx(smiles)
        
        one_hot = {edge: torch.tensor(self.edge_map[label]) for edge, label in
                   (nx.get_edge_attributes(graph, 'bond_type')).items()}
        nx.set_edge_attributes(graph, name='one_hot', values=one_hot)
        
        at_type = {a: torch.tensor(self.at_map[label]) for a, label in
                   (nx.get_node_attributes(graph, 'atomic_num')).items()}
        nx.set_node_attributes(graph, name='atomic_num', values=at_type)
        
        at_charge = {a: torch.tensor(self.charges_map[label]) for a, label in
                   (nx.get_node_attributes(graph, 'formal_charge')).items()}
        nx.set_node_attributes(graph, name='formal_charge', values=at_charge)
        
        at_chir = {a: torch.tensor(self.chi_map[label]) for a, label in
                   (nx.get_node_attributes(graph, 'chiral_tag')).items()}
        nx.set_node_attributes(graph, name='chiral_tag', values=at_chir)
        
        # Create dgl graph
        g_dgl = dgl.DGLGraph()
        g_dgl.from_networkx(nx_graph=graph, 
                            node_attrs=['atomic_num','chiral_tag','formal_charge'],
                            edge_attrs=['one_hot'])

        n_nodes = len(g_dgl.nodes())
        g_dgl.ndata['h'] = torch.ones((n_nodes, self.emb_size)) # nodes embeddings 
        
        return g_dgl, targets
        
    
class Loader():
    def __init__(self,
                 csv_path,
                 n_mols,
                 emb_size=1,
                 batch_size=64,
                 num_workers=20,
                 debug=False,
                 shuffled=False):
        """
        Wrapper for test loader, train loader 
        Uncomment to add validation loader 

        """

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = molDataset(csv_path, n_mols,
                                  emb_size,
                                  debug=debug,
                                  shuffled=shuffled)
        self.num_edge_types = self.dataset.num_edge_types
        
    def get_maps(self):
        # Returns dataset mapping of edge and node features 
        return self.dataset.edge_map, self.dataset.at_map, self.dataset.chi_map, self.dataset.charges_map
    
    def get_reverse_maps(self):
        # Returns maps of one-hot index to actual feature 
        rev_em = {v:i for (i,v) in self.dataset.edge_map.items() }
        rev_am = {v:i for (i,v) in self.dataset.at_map.items() }
        rev_chi_m={v:i for (i,v) in self.dataset.chi_map.items() }
        rev_cm={v:i for (i,v) in self.dataset.charges_map.items() }
        return rev_em, rev_am, rev_chi_m, rev_cm

    def get_data(self):
        n = len(self.dataset)
        print(f"Splitting dataset with {n} samples")
        indices = list(range(n))
        # np.random.shuffle(indices)
        np.random.seed(0)
        split_train, split_valid = 0.8, 0.8
        train_index, valid_index = int(split_train * n), int(split_valid * n)
        
        train_indices = indices[:train_index]
        valid_indices = indices[train_index:valid_index]
        test_indices = indices[valid_index:]
        
        train_set = Subset(self.dataset, train_indices)
        valid_set = Subset(self.dataset, valid_indices)
        test_set = Subset(self.dataset, test_indices)
        print(f"Train set contains {len(train_set)} samples")


        train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                  num_workers=self.num_workers, collate_fn=collate_block)

        # valid_loader = DataLoader(dataset=valid_set, shuffle=True, batch_size=self.batch_size,
        #                           num_workers=self.num_workers, collate_fn=collate_block)
        
        test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=self.batch_size,
                                 num_workers=self.num_workers, collate_fn=collate_block)


        # return train_loader, valid_loader, test_loader
        return train_loader, 0, test_loader
    
if(__name__=='__main__'):
    d=molDataset()