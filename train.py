# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019

@author: jacqu

RGCN model to predict molecular LogP 

"""
import sys
import torch
import dgl
import pickle
import torch.utils.data
from torch import nn, optim
import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F

if (__name__ == "__main__"):
    sys.path.append("./dataloading")
    from rgcn import Model
    from molDataset import molDataset, Loader
    from utils import *
    from viz import *
    from draw_mol import *
    from rdkit_to_nx import *
    
    # config
    n_epochs = 100 # epochs to train
    batch_size = 128
    display_test=False
    SAVE_FILENAME='./saved_model_w/logp.pth'
    model_path= 'saved_model_w/logp.pth'
    log_path='./saved_model_w/logs_logp.npy'
    
    load_model = False

    
    #Load train set and test set
    loaders = Loader(csv_path='data/moses_train.csv',
                     n_mols=1000,
                     num_workers=0, 
                     batch_size=batch_size, 
                     shuffled= True,
                     target = 'logP')
    rem, ram, rchim, rcham = loaders.get_reverse_maps()
    
    train_loader, _, test_loader = loaders.get_data()
    
    # Logs
    logs_dict = {'train_mse':[],
                 'test_mse':[]}
    
    #Model & hparams
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parallel=False
    params ={'atom_types':loaders.num_atom_types, #node embedding dimension
             'charges': loaders.num_charges,
             'h_dim':16,
             'out_dim':4,
             'num_rels':loaders.num_edge_types,
             'num_bases' :-1,
             'num_hidden_layers':2}
    pickle.dump(params, open('saved_model_w/params.pickle','wb'))

    model = Model(**params).to(device)
    if(load_model):
        model.load_state_dict(torch.load(model_path))
    
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
    model.train()
    for epoch in range(1, n_epochs+1):
        print(f'Starting epoch {epoch}')
        epoch_loss=0

        for batch_idx, (graph, target) in enumerate(train_loader):
        
            target=target.to(device).view(-1,1) # Graph-level target : (batch_size,)
            # Embedding for each node
            graph=send_graph_to_device(graph,device)
            out = model(graph).view(-1,1)
            
            # print(out.shape)
            
            #Compute loss : change according to supervision 
            t_loss=F.mse_loss(out,target,reduction='sum')
            epoch_loss+=t_loss.item()
            
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
                
                target=target.to(device).view(-1,1) # Graph-level target : (batch_size,)
                graph=send_graph_to_device(graph,device)
                out=model(graph).view(-1,1)
                
                t_loss+=F.mse_loss(out,target,reduction='sum')
            
                
                # Try out attention
                if(batch_idx==0 and display_test):
                    # Att has shape h, dest_nodes, src_nodes
                    # Sum of attention[1]=1 (attn weights sum to one for destination node)
                    print([(p,t) for (p,t) in zip(list(out.cpu().numpy()),list(target.cpu().numpy())) ] )
                
                
            print(f'Validation loss at epoch {epoch}, per batch: {t_loss/len(test_loader)}')
            logs_dict['test_mse'].append(t_loss/len(test_loader))
            logs_dict['train_mse'].append(epoch_loss/len(train_loader))
            
        if(epoch%5==0):
            #Save model : checkpoint      
            torch.save( model.state_dict(), SAVE_FILENAME)
            pickle.dump(logs_dict, open(log_path,'wb'))
            print(f"model saved to {SAVE_FILENAME}")
        