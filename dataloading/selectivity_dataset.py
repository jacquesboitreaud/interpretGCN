# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 08:07:32 2019

@author: jacqu
"""

import pandas as pd 

df = pd.read_csv('../data/test_set.csv')

high = df[df['logP']>4.5]
low = df[df['logP']<-1]

high.to_csv('../data/high_logp.csv')

low.to_csv('../data/low_logp.csv')