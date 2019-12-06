# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:52:08 2019

@author: jacqu

Plot training and test prediction error for HERG affinity prediction 
"""

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

logs = '../saved_model_w/logs_herg.npy'

dic = np.load(logs, allow_pickle=True)

train, test = dic['train_bce'], dic['test_bce']

# pb with type of test values
test = [t.item() for t in test]


# Visualize
t = np.arange(len(train))
sns.lineplot(t, train, label='train')
sns.lineplot(t, test, label='test')
#plt.title('Average loss per batch during training')
plt.xlabel('Epochs')
plt.ylabel('Reconstruction loss per item ')
plt.legend()
