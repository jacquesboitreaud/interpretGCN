# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 15:41:09 2019

@author: jacqu

Enrichment factor at different thresholds 
"""
import numpy as np

percentage = 20

def frac_polars(s):
    n=0
    m=Chem.MolFromSmiles(s)
    tot = m.GetNumAtoms()
    for c in s:
        if(c in {'O','F','N','Cl','Br','I'}):
            n+=1
    return n/tot

def num_polars(s):
    n=0
    m=Chem.MolFromSmiles(s)
    tot = m.GetNumAtoms()
    for c in s:
        if(c in {'O','F','N','Cl','Br','I'}):
            n+=1
    return n, tot

        

subs = [kv[0] for kv in std]
freq = [kv[1] for kv in std]
rank1 = [i for i in range(len(std))]
n_polar = [num_polars(s[0])[0] for s in std]
n_tot = [num_polars(s[0])[1] for s in std]

df = pd.DataFrame.from_dict({'s':subs,
                             'attn_freq':freq,
                             'r1':rank1,
                             'polars':n_polar,
                             'tot': n_tot})

df=df.sort_values('attn_freq')
overall = np.sum(df['polars'])/np.sum(df['tot'])

k=int(percentage/100*74)
enrichment_k = np.sum(df['polars'][-k:])/(np.sum(df['tot'][-k:])*overall)
print(f'Enrichment at {percentage}% is ', enrichment_k)