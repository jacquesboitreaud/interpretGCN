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

df = pd.read_csv('../data/HERG_2classes.csv')

b=list(df['pIC50'])
b= [int(x>0) for x in b]

df['binary']=b

df.to_csv('../data/HERG_2classes.csv')