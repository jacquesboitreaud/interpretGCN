# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 13:45:26 2019

@author: jacqu

Statistics on atom and substructure contributions obtained with Integrated Gradients method
Run it after running integrad_logp.py 
"""

if(__name__=='__main__'):
    
    N_mols=5000
    batch_size=100
    # List all substructures
    with open('data/vocab.txt','r') as f:
        vocab = f.readlines()
        vocab = [s.rstrip('\n') for s in vocab]
        
    ctr = {v: [] for v in vocab} # dict of lists 
    occur = {v: 0 for v in vocab}
    # Get vocabulary of substructures 
    vocab_dict = {s:i for (i,s) in enumerate(vocab)}
    
    # Load model, pass molecules, get attributions, compute zscores
    loader = Loader(csv_path='data/test_set.csv',
                 n_mols=N_mols,
                 num_workers=0, 
                 batch_size=batch_size, 
                 shuffled= True,
                 target = 'logP',
                 test_only=True)
    rem, ram, rchim, rcham = loader.get_reverse_maps()
    _ ,_ , test_loader = loader.get_data()
    
    # # Instantiate IG + load model 
    model_path= 'saved_model_w/logp_lowd.pth'
    params = pickle.load(open('saved_model_w/params.pickle','rb'))
    params['classifier']=False
    params['features_dim']=14
    
    model = Model(**params)
    model.load_state_dict(torch.load(model_path))
    inteGrad = IntegratedGradients(model)
    
    nodes = -1
    for i, (graph, targets) in enumerate(test_loader):
        print(i)
        graphs = dgl.unbatch(graph)
        for m in range(batch_size):
            x, target = graphs[m], targets[m]
            attrib, _ , delta = inteGrad.attrib(x, nodes)
            
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
            
            try:
                mol = nx_to_mol(x, rem, ram, rchim, rcham)
        
                for v in vocab: 
                    # Get atoms that match substructure 
                    matches = set(mol.GetSubstructMatch(Chem.MolFromSmiles(v)))
                    if(len(matches)>0):
                        occur[v]+=1 # substructure occurence
                        # Summing up atom contributions 
                        c_contribs=[node_contribs['charge'][i] for i in matches]
                        a_contribs=[node_contribs['atom type'][i] for i in matches]
                        
                        ctr[v].append(sum(a_contribs)+sum(c_contribs))
            except:
                continue
            
    # LOOK AT SUBSTRUCTURES:
    ctr = {k:v for (k,v) in ctr.items() if occur[k]>0}
    np.save('stats_dict.npy',ctr)
    means = {k:np.mean(v)/num_atoms(k) for (k,v) in ctr.items()}
    
    # Non zero freqs : 
    std = sorted(means.items(),key=lambda x: x[1])
    
    sns.barplot(x=np.arange(len(std)), y=[kv[1] for kv in std])
    
    highs, lows = std[-25:], std[:25]

    fig=plt.figure(dpi=300, figsize=(20,20))
    img=draw_multi([g[0] for g in highs])
    img2=draw_multi([g[0] for g in lows])
        