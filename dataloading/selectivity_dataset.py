# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:02:53 2019

@author: jacqu

Build specific dataset for precise tasks:
    - compound selectivity between 2 targets 
    - predicting binding affinity for compounds assayed in CHEMBL 
"""

import pandas as pd 
import numpy as np

df = pd.read_csv('../data/HERG_dataset.csv')

df2 = pd.read_csv('../data/CHEMBL_18t.csv')
df2=df2[df2['HERG']<=0]
"""
df=df[df['HERG']>0]

df=df.loc[:,['can','HERG']]

max(df['HERG'])

df['pIC50']=-np.log(1e-9*df['HERG'])

df.to_csv('../HERG_dataset.csv')
"""