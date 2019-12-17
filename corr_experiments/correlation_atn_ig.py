# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:38:33 2019

@author: jacqu
"""

attn = std

ig_dict = means

d={'s':[],
   'a':[],
   'ig':[]}

for kv in attn : 
    s = kv[0]
    if(s in ig_dict.keys() and ('O' in s or 'N' in s or 'Cl' in s)):
        d['s'].append(s)
        d['ig'].append(np.mean(ig_dict[s]))
        d['a'].append(float(kv[1]))
        
df=pd.DataFrame.from_dict(d)
        
sns.scatterplot(df['ig'], df['a'])

df.to_csv('all_logp_corr.csv')

# Spearman correlaton
df['rank_a']=pd.Series(np.arange(len(d['s'])),index=df.index)

df=df.sort_values('ig')

df['rank_ig']=pd.Series(np.arange(len(d['s'])),index=df.index)

df['dsquared']=(df['rank_ig']-df['rank_a'])**2

# Spearman coeff :
rho = 1 - 6*np.sum(df['dsquared'])/(38*(38**2-1))
print(rho)

sns.scatterplot(df['ig'], df['a'])
