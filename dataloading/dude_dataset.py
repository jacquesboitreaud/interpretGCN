# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:02:53 2019

@author: jacqu

Build specific dataset for DUDE task
"""

import pandas as pd 
import numpy as np

d = {'can':[],'binary':[]}

with open('C:/Users/jacqu/Documents/mol2_resource/dud/all/esr1/actives_final.ism','r') as f:
   line = f.readline()
   while line:
       smi, _,_ = line.split()
       print(smi)
       line = f.readline()
       d['can'].append(smi)
       d['binary'].append(1)
       
with open('C:/Users/jacqu/Documents/mol2_resource/dud/all/esr1/decoys_final.ism','r') as f:
   line = f.readline()
   while line:
       smi, _ = line.split()
       print(smi)
       line = f.readline()
       d['can'].append(smi)
       d['binary'].append(0)

df=pd.DataFrame.from_dict(d)
df.to_csv('../data/ESR1_2classes.csv')