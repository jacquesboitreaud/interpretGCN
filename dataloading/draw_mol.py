# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:04:26 2019

@author: jacqu

Draw molecules and highlight atoms 
"""

import dgl

import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import ChemicalFeatures
from rdkit import rdBase
from rdkit.RDPaths import RDDocsDir
from rdkit.RDPaths import RDDataDir
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import os
from IPython.display import SVG
print(rdBase.rdkitVersion)
IPythonConsole.ipython_useSVG=False

if (__name__ == "__main__"):
    sys.path.append("./dataloading")
    from rdkit_to_nx import *
    
    
def highlight(mol, atidxs,labels):
    highlighted = list(atidxs)
    # Colors for highlight
    colors=[{i:(1,0,0) for i in highlighted[0]}, {i:(0.67,0.84,0.9) for i in highlighted[1]}]
    
    drawer = rdMolDraw2D.MolDraw2DSVG(800,300,400,300)
    opts = drawer.drawOptions()

    for i in range(mol.GetNumAtoms()):
        opts.atomLabels[i] = mol.GetAtomWithIdx(i).GetSymbol()+str(i)
    AllChem.Compute2DCoords(mol)
    

    drawer.DrawMolecules([mol]*2, legends=labels, highlightAtoms=highlighted,
                               highlightAtomColors=colors, highlightBonds=[[],[]])#, highlightBondColors=colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:','')
    SVG(svg)
    
    return svg

def highlight_noid(mol, atidxs,labels):
    highlighted = list(atidxs)
    # Colors for highlight
    colors={i:(1,0,0) for i in highlighted[0]}, {i:(0.2,0,1) for i in highlighted[1]}
    
    drawer = rdMolDraw2D.MolDraw2DSVG(800,300,400,300)

    AllChem.Compute2DCoords(mol)
    

    drawer.DrawMolecules([mol]*2, legends=labels, highlightAtoms=highlighted,
                               highlightAtomColors=colors, highlightBonds=[[],[]])#, highlightBondColors=colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:','')
    SVG(svg)
    
    return svg

def highlight_att(mol, atidxs,labels):
    highlighted = list(atidxs)
    # Colors for highlight
    colors={i:(1,0.5,0) for i in highlighted} 
    
    drawer = rdMolDraw2D.MolDraw2DSVG(400,400)
    opts = drawer.drawOptions()

    for i in range(mol.GetNumAtoms()):
        opts.atomLabels[i] = mol.GetAtomWithIdx(i).GetSymbol()+str(i)
    AllChem.Compute2DCoords(mol)
    

    drawer.DrawMolecule(mol, highlightAtoms=highlighted,
                               highlightAtomColors=colors, highlightBonds=[])#, highlightBondColors=colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:','')
    SVG(svg)
    
    return svg

def highlight_att_noid(mol, atidxs,labels):
    highlighted = list(atidxs)
    # Colors for highlight
    colors={i:(1,0.5,0) for i in highlighted} 
    
    drawer = rdMolDraw2D.MolDraw2DSVG(400,400)

    AllChem.Compute2DCoords(mol)
    
    drawer.DrawMolecule(mol, highlightAtoms=highlighted,
                               highlightAtomColors=colors, highlightBonds=[])#, highlightBondColors=colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:','')
    SVG(svg)
    
    return svg


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
