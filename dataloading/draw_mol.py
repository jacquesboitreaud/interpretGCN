# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:04:26 2019

@author: jacqu

Draw molecules and highlight atoms 
"""

import dgl

import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import rdBase
from rdkit.RDPaths import RDDocsDir
from rdkit.RDPaths import RDDataDir
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import os
print(rdBase.rdkitVersion)
IPythonConsole.ipython_useSVG=True

if (__name__ == "__main__"):
    sys.path.append("./dataloading")
    from rdkit_to_nx import *
    
    
def highlight(mol, atidxs,labels):
    #color : RGB tuple, size 3
    highlighted = list(atidxs)
    # Colors for highlight
    colors=[{i:(1,0,0) for i in highlighted[0]}, {i:(0.2,0,1) for i in highlighted[1]}]
    AllChem.Compute2DCoords(mol)
    plt.figure(figsize = (10,4)) # w*h
    img = Draw.MolsToGridImage([mol]*2, legends=labels, highlightAtomLists=highlighted,
                               highlightAtomColors=colors)#, highlightBondColors=colors)
    return img

def draw_smi(smiles):
    mol=Chem.MolFromSmiles(smiles)
    img = Chem.Draw.MolToImage(mol)
    plt.imshow(img)
    plt.show()
    return img

def draw_multi(smiles):
    # list of smiles 
    mols=[Chem.MolFromSmiles(s) for s in smiles]
    img = Draw.MolsToGridImage(mols, molsPerRow=7,maxMols=75, subImgSize=(100, 100), legends=[str(i) for i in range(len(mols))])
    return img

def num_atoms(s):
    # smiles as input
    m=Chem.MolFromSmiles(s)
    return m.GetNumAtoms()
