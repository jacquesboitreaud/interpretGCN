# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:04:26 2019

@author: jacqu

Draw molecules and highlight atoms 
"""

import dgl

import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG, Image
IPythonConsole.molSize = (400,400)

if (__name__ == "__main__"):
    sys.path.append("./dataloading")
    from rdkit_to_nx import *
    
    
def highlight(mol, atidxs):
    # Prints a depiction of molecule object with list of atoms highlighted 
    # Can also be done for a list of bonds
    highlighted = atidxs
    plt.figure(figsize = (10,4)) # w*h
    img = Chem.Draw.MolToImage(mol, highlightAtoms=atidxs, highlightColor=[1,0.7,0]) # highlight in orange
    plt.imshow(img)
    plt.show()
    return img

def draw_smi(smiles):
    mol=Chem.MolFromSmiles(smiles)
    img = Chem.Draw.MolToImage(mol)
    plt.imshow(img)
    plt.show()
    return img

