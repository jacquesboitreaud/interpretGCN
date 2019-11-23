# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:02:53 2019

@author: jacqu

Build specific dataset for compound selectivity
"""

import pandas as pd 

df = pd.read_csv('../../data/CHEMBL_18t.csv', nrows=100)

print(df.columns)