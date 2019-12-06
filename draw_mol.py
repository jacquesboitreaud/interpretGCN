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
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem.Draw import DrawingOptions

DrawingOptions.atomLabelFontSize = 55
DrawingOptions.dotsPerAngstrom = 2000
DrawingOptions.bondLineWidth = 2


from IPython.display import SVG, Image
IPythonConsole.molSize = (800,800)

if (__name__ == "__main__"):
    sys.path.append("./dataloading")
    from rdkit_to_nx import *
    
    
def highlight(mol, atidxs, color=[1,0.7,0]):
    # Prints a depiction of molecule object with list of atoms highlighted 
    # Can also be done for a list of bonds
    #color : RGB tuple, size 3
    highlighted = atidxs
    plt.figure(figsize = (10,4)) # w*h
    img = Chem.Draw.MolToImage(mol, highlightAtoms=atidxs, highlightColor=color) # highlight in orange
    plt.imshow(img)
    plt.show()
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
    img = MolsToGridImage(mols, molsPerRow=7,maxMols=60, subImgSize=(100, 100), legends=[str(i) for i in range(len(mols))])
    return img

def num_atoms(s):
    # smiles as input
    m=Chem.MolFromSmiles(s)
    return m.GetNumAtoms()
