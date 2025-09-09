import subprocess
# subprocess.check_call(["pip", "install", "gensim"])
from torch.utils.data import DataLoader, TensorDataset, random_split
from gensim.models    import Word2Vec

import torch.nn.functional as F
import torch.nn as nn
import argparse
import logging
import torch
import boto3
import math
import copy
import os

from tqdm import tqdm

torch.manual_seed(42)

def FedAvg(global_model: dict, client_state_dicts: list[dict]) -> nn.Module:
    for key in global_model.keys():
        stacked = torch.stack([client_dict[key].float() for client_dict in client_state_dicts], dim=0)
        global_model[key] = torch.mean(stacked, dim=0)
    return global_model

def client_update(model : nn.Module , 
                  train_loader: DataLoader, 
                  epochs: int = 1, lr: float = 0.1, device : str = 'cpu') -> dict:
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas =  (0.9,0.99)) #(0.1,0.3)
    loss_fn   = nn.CrossEntropyLoss()

    for _ in (range(epochs)):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds, alpha = model(X_batch)
            loss = loss_fn(preds, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model.state_dict()


class ConvAttnPool(nn.Module):
    def __init__(self, table_path,label_space = 50, num_of_filters = 10, kernel_size = 3, drop_out = 0.2):
        super().__init__()
        model = Word2Vec.load(table_path)
        vocab_size, embed_d = model.wv.vectors.shape
        embed_table = torch.from_numpy(model.wv.vectors).type(torch.float32)
        embed_table = torch.concat([embed_table,torch.zeros(size = (1,embed_d))], dim = 0)

        self.embed = nn.Embedding.from_pretrained(embeddings=embed_table,padding_idx=vocab_size)
        self.conv  = nn.Conv1d(embed_d, num_of_filters, kernel_size = kernel_size, padding = int(math.floor(kernel_size//2)))
        self.U     = nn.Linear(num_of_filters, label_space)
        self.final = nn.Linear(num_of_filters, label_space)
        self.embed_drop     = nn.Dropout(p = drop_out)
        self.embedding_size = embed_d

    def forward(self, x):
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose (1, 2)
        x = F. tanh(self.conv(x).transpose(1,2))
        alpha = F.softmax(self.U.weight.matmul(x.transpose (1,2)) , dim=2)
        m     = alpha.matmul(x)
        y     = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        return y, alpha
    
def GenerateModel(table_path, num_of_filters = 15,kernel_size = 5):
    return ConvAttnPool(
            table_path     = table_path,
            drop_out       = 0.2,
            num_of_filters = num_of_filters,
            label_space    = 50, 
            kernel_size    = kernel_size
            )

def federate_model(config: dict):
    """
    config
        batch_size
        lr
        n_filters
        window_size
        epochs
        rounds
    """
    # curr_dir = os.path.split(os.getcwd())[0]
    curr_dir = "/home/drew/FL-with-MIMIC/Replicating Mullenbach/AWS"
    model_param_path = os.path.join(curr_dir,"Model",'processed_full.w2v')
    # Training Data
    X_train = torch.load(os.path.join(curr_dir,"Data","X_train.pt"))
    Y_train = torch.load(os.path.join(curr_dir,"Data","Y_train.pt"))
    ################################## Split the dataset three ways
    train = TensorDataset(X_train,Y_train)

    c1, c2, c3 = random_split(train,lengths=[0.33,0.33,0.34])
    c1_loader  = DataLoader(c1, batch_size = config['batch_size'], shuffle = True)
    c2_loader  = DataLoader(c2, batch_size = config['batch_size'], shuffle = True)
    c3_loader  = DataLoader(c3, batch_size = config['batch_size'], shuffle = True)
    c_loaders  = [c1_loader, c2_loader, c3_loader]
    ##################################
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ################################## Federated Training
    Global_Model = GenerateModel(table_path     = model_param_path,
                                 num_of_filters = config['n_filters'], 
                                 kernel_size    = config['window_size'])
    for rnds in range(config['rounds']):
        client = GenerateModel(table_path     = model_param_path,
                               num_of_filters = config['n_filters'], 
                               kernel_size    = config['window_size'])
        client.load_state_dict(Global_Model.state_dict())
        c_parameters = []
        for c_idx in range(3): # Static since we have n = 3 clients
            c_param = client_update(model  = client,
                                    epochs = config['epochs'],
                                    lr     = config['lr'],
                                    device = device,
                                    train_loader = c_loaders[c_idx])
            c_parameters.append(c_param)
        new_param = FedAvg(Global_Model.state_dict(), c_parameters)
        Global_Model.load_state_dict(new_param)
    ################################
    return Global_Model.state_dict()
    
if __name__ == '__main__':
    config = {
        "batch_size" : 16,
        "lr"         : 0.0001,
        "n_filters"  : 5,
        "window_size": 3,
        "epochs"     : 2,
        "rounds"     : 5
    }
    federate_model(config)
    exit()
    ################################### Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs",type=int  , default=2)
    parser.add_argument("--rounds"    ,type=int  , default=5)
    parser.add_argument("--batch_size",type=int  , default=32)
    parser.add_argument("--filters"   ,type=int  , default=10)
    parser.add_argument("--window"    ,type=int  , default=3)
    parser.add_argument("--lr"        ,type=float, default=0.01)
    parser.add_argument("--s3_path"   ,type=str)

    # parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    args   = parser.parse_args()
    s3_bucket = 'sagemaker-us-east-1-762568382963'
    ################################### Logging
    curr_dir = os.getcwd()
    ################################## Download textual embeddings
    if not os.path.isfile(os.path.join(curr_dir,'Model_Param/processed_full.w2v')):
        print("Downloading Model Parameters!")
        s3 = boto3.client('s3')
        s3.download_file(s3_bucket,
                         'model/processed_full.w2v',
                         os.path.join(curr_dir,'Model_Param/processed_full.w2v'))
        s3.download_file(s3_bucket,
                         'model/processed_full.w2v.syn1neg.npy',
                         os.path.join(curr_dir,'Model_Param/processed_full.w2v.syn1neg.npy'))
        s3.download_file(s3_bucket,
                         'model/processed_full.w2v.wv.vectors.npy',
                         os.path.join(curr_dir,'Model_Param/processed_full.w2v.wv.vectors.npy'))
    ################################## Download our dataset for training
    if not os.path.isfile(os.path.join(curr_dir,'Data/X_train.pt')):
        print("Downloading Training Dataset!")
        s3 = boto3.client('s3')
        s3.download_file(s3_bucket,
                         'Data/X_train.pt',
                         os.path.join(curr_dir,'Data/X_train.pt'))
        s3.download_file(s3_bucket,
                         'Data/Y_train.pt',
                         os.path.join(curr_dir,'Data/Y_train.pt'))
    ################################## Download our dataset for training
    model_param_path = os.path.join(curr_dir,"Model_Param",'processed_full.w2v')
    # Training Data
    X_train = torch.load(os.path.join(curr_dir,"Data","X_train.pt"))
    Y_train = torch.load(os.path.join(curr_dir,"Data","Y_train.pt"))
    ################################## Split the dataset three ways
    train = TensorDataset(X_train,Y_train)

    c1, c2, c3 = random_split(train,lengths=[0.33,0.33,0.34])
    c1_loader  = DataLoader(c1, batch_size = args.batch_size, shuffle = True)
    c2_loader  = DataLoader(c2, batch_size = args.batch_size, shuffle = True)
    c3_loader  = DataLoader(c3, batch_size = args.batch_size, shuffle = True)
    c_loaders  = [c1_loader, c2_loader, c3_loader]
    ##################################
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ################################## Federated Training
    Global_Model = GenerateModel(table_path = model_param_path,
                            num_of_filters = args.filters, 
                            kernel_size    = args.window)
    for rnds in range(args.rounds):
        client = GenerateModel(table_path  = model_param_path,
                            num_of_filters = args.filters, 
                            kernel_size    = args.window)
        client.load_state_dict(Global_Model.state_dict())
        c_parameters = []
        for c_idx in range(3): # Static since we have n = 3 clients
            c_param = client_update(model  = client,
                                    epochs = args.num_epochs,
                                    lr     = args.lr,
                                    device = device,
                                    train_loader = c_loaders[c_idx])
            c_parameters.append(c_param)
        new_param = FedAvg(Global_Model.state_dict(), c_parameters)
        Global_Model.load_state_dict(new_param)
    ################################## Federated Training
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(Global_Model.state_dict(), f)

    print("Done Training!")
                                         
        



    


    