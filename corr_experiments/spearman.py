# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 15:41:09 2019

@author: jacqu
"""
import numpy as np
from rdkit.Chem import Descriptors

def frac_polars(s):
    n=0
    m=Chem.MolFromSmiles(s)
    logP=Descriptors.MolLogP(m)
    tot = m.GetNumAtoms()
    for c in s:
        if(c in {'O','F','N','Cl','Br','I'}):
            n+=1
    return n/tot, logP

def frac_polars(s):
    n=0
    m=Chem.MolFromSmiles(s)
    val=Descriptors.NumValenceElectrons(m)
    tot = m.GetNumAtoms()
    for c in s:
        if(c in {'O','F','N','Cl','Br','I'}):
            n+=1
    return n/tot, val
        

subs = [kv[0] for kv in std]
freq = [kv[1] for kv in std]
rank1 = [i for i in range(len(std))]
frac_polar = [frac_polars(s[0])[1] for s in std]

df = pd.DataFrame.from_dict({'s':subs,
                             'ig':freq,
                             'r1':rank1,
                             'polar_freq':frac_polar})
    
df = df.sort_values('polar_freq')
df['r2']=pd.Series(np.arange(len(subs)), index=df.index)

df['dsquared']=(df['r1']-df['r2'])**2

# Spearman coeff :
rho = 1 - 6*np.sum(df['dsquared'])/(86*(86**2-1))

sns.scatterplot(df['polar_freq'], df['ig'], color='orange')