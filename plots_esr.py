# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 08:54:49 2019

@author: jacqu
"""

m = 40
nodes = -1


graph, target = next(iter(test_loader))
graphs = dgl.unbatch(graph)
x, target = graphs[m], target[m]
out = model(graph)[m]
print(f'Predicted binding proba is {out.item()}, true is {target.item()}')
attrib, _ , delta = inteGrad.attrib(x, nodes)

# Attrib to dataframe 
df = pd.DataFrame(attrib.numpy())
df.columns = ['charge 0', 'charge +1', 'charge -1', 'H','Br','C','N','O','F','P','S','Cl','I']


print(torch.sum(attrib).item(), delta)
sns.heatmap(df.transpose(), vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
plt.xlabel('Node n°')




# Select + and - edges (atoms):
x.ndata['ig']=attrib # add attributions as a node feature
x=x.to_networkx(node_attrs=['atomic_num','chiral_tag','formal_charge','num_explicit_hs','is_aromatic','ig'], 
                    edge_attrs=['one_hot'])
x=x.to_undirected()
node_contribs = {'atom type':[], 'charge':[]}
for (i, data) in x.nodes(data=True):
    at_charge, at_type = torch.argmax(data['formal_charge']).item(), torch.argmax(data['atomic_num']).item()
    node_contribs['charge'].append(data['ig'][at_charge].item())
    node_contribs['atom type'].append(data['ig'][4+at_type].item())
        
# Highlighting : Relative to other contributions 
"""
mean_t, mean_c = np.mean(node_contribs['atom type']), np.mean(node_contribs['charge'])
sd_t, sd_c = np.std(node_contribs['atom type']), np.std(node_contribs['charge'])

z_t= (node_contribs['atom type']-mean_t)/sd_t
z_c= (node_contribs['charge']-mean_c)/sd_c
"""
z_t=np.array(node_contribs['atom type'])
z_c=np.array(node_contribs['charge'])

pos = list(np.where(z_t>0)[0])
neg = list(np.where(z_t<0)[0])

# Plot heatmap of node contributions: 
plt.figure()
df2=pd.DataFrame.from_dict({'atom type':z_t, 'charge':z_c})
sns.heatmap(df2.transpose(), annot=False, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
plt.xlabel('Atom n°')

# For molecule plot with highlights, 
mol=nx_to_mol(x,rem, ram, rchim, rcham )
# To networkx and plot with colored bonds 
labels=['+','-']
img =highlight(mol,(tuple(pos),tuple(neg)),labels)