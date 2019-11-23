# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019

@author: jacqu

RGCN model to predict molecular LogP 

"""
import sys
import torch
import dgl
import torch.utils.data
from torch import nn, optim
import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F

if (__name__ == "__main__"):
    sys.path.append("./dataloading")
    from rgcn import Model, simLoss
    from molDataset import molDataset, Loader
    from utils import *
    from viz import *
    from draw_mol import *
    
    # config
    N=1 # num node features 
    N_types=44
    n_hidden = 16 # number of hidden units
    n_bases = -1 # use number of relations as number of bases
    n_hidden_layers = 1 # use 1 input layer, 1 output layer, no hidden layer
    n_epochs = 10 # epochs to train
    batch_size = 40
    
    #Load train set and test set
    loaders = Loader(csv_path='../data/CHEMBL_18t.csv',
                     n_mols=100000,
                     num_workers=4, 
                     batch_size=batch_size, 
                     shuffled= True)
    rem, ram, rchim, rcham = loaders.get_reverse_maps()
    
    train_loader, _, test_loader = loaders.get_data()
    
    #Model & hparams
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parallel=False

    model = Model(num_nodes=N, h_dim=16, out_dim=1, num_rels=N_types, num_bases=-1).to(device)
    
    if (parallel): #torch.cuda.device_count() > 1 and
        print("Start training using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        
    #Print model summary
    print(model)
    map = ('cpu' if device == 'cpu' else None)
    torch.manual_seed(1)
    optimizer = optim.Adam(model.parameters())
    #optimizer = optim.Adam(model.parameters(),lr=1e-4, weight_decay=1e-5)
    
    #Train & test
    for epoch in range(1, n_epochs+1):
        print(f'Starting epoch {epoch}')
        model.train()
        for batch_idx, (graph, target) in enumerate(train_loader):
        
            target=target.to(device).view(-1) # Graph-level target : (batch_size,)
            # Embedding for each node
            graph=send_graph_to_device(graph,device)
            out = model(graph).squeeze()
            
            # print(out.shape)
            
            #Compute loss : change according to supervision 
            t_loss=F.mse_loss(out,target,reduction='sum')
            
            # backward loss 
            optimizer.zero_grad()
            t_loss.backward()
            #clip.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            
            #logs and monitoring
            if batch_idx % 100 == 0:
                # log
                print('ep {}, batch {}, loss : {:.2f} '.format(epoch, 
                      batch_idx, t_loss.item()))
        
        # Validation pass
        model.eval()
        t_loss = 0
        with torch.no_grad():
            for batch_idx, (graph, target) in enumerate(test_loader):
                
                target=target.to(device).view(-1) # Graph-level target : (batch_size,)
                graph=send_graph_to_device(graph,device)
                out=model(graph).squeeze()
                
                t_loss=F.mse_loss(out,target,reduction='sum')
                
                # Try out attention
                if(batch_idx==0):
                    att= get_attention_map(graph, src_nodes=graph.nodes(), dst_nodes=graph.nodes(), h=1)
                    # Att has shape h, dest_nodes, src_nodes
                    # Sum of attention[1]=1 (attn weights sum to one for destination node)
                    
                    # Transform graph to RDKit molecule for nice visualization
                    graphs = dgl.unbatch(graph)
                    g0=graphs[0]
                    n_nodes = len(g0.nodes)
                    att_g0 = att[0,:n_nodes,:n_nodes] # get attn weights only for g0
                    
                    # Select atoms with highest attention weights and plot them 
                    tops = np.unique(np.where(att_g0>0.55)) # get top atoms in attention
                    mol = nx_to_mol(g0, rem, ram, rchim, rcham)
                    img=highlight(mol,list(tops))
                
                
                
                
            print(f'Validation loss at epoch {epoch}: {t_loss}')
        