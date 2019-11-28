# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:08:23 2019

@author: jacqu

Uses RDKit to find common substructures in molecules 
"""

from rdkit import Chem
from rdkit.Chem import rdFMCS

with open('../data/vocab.txt','r') as f:
    vocab = f.readlines()
    
# Get vocabulary of substructures 
vocab_dict = {s.rstrip('\n'):i for (i,s) in enumerate(vocab)}

#mol1.GetSubstructMatch(mol2)

