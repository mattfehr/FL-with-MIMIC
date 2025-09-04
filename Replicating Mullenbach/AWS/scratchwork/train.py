from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics  import classification_report
from sklearn.manifold import TSNE
from tqdm             import tqdm

import matplotlib.pyplot   as plt
import torch.nn as nn
import argparse
import torch

def train_model(model,device, optimizer ,data_loader, loss_module, num_epochs=100):
    model.train()
    train_losses  = []
    for epoch in (range(num_epochs)):
        for data_inputs, data_labels in data_loader:
            # Put training data into GPU
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            preds = model(data_inputs).to(device)
            loss  = loss_module(preds, data_labels)
            train_losses.append(loss.item())

            # Zero the gradients
            optimizer.zero_grad()
            loss     .backward()
            optimizer.step()

    return train_losses

def generate_model():
    model = nn.Sequential(
        nn.Linear(10, 100),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(100, 3),
        nn.Sigmoid()
    )
    return model

def generate_data():
    X, Y = make_multilabel_classification(n_features      = 10,
                                      n_samples       = 100,
                                      length          = 20,
                                      n_classes       = 3,
                                      n_labels        = 1,
                                      allow_unlabeled = False,
                                      random_state    = 42)
    X = torch.from_numpy(X).type(torch.float32)
    Y = torch.from_numpy(Y).type(torch.float32)


    train, val, test = random_split(TensorDataset(X,Y), lengths=[0.6,0.2,0.2])

    train_loader = DataLoader(train , batch_size = 10, shuffle = True)
    val_loader   = DataLoader(val   , batch_size = 10, shuffle = True)
    test_loader  = DataLoader(test  , batch_size = 10, shuffle = True)
    return train_loader,val_loader, test_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_epochs", type=int, default=10
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )

    args = parser.parse_args()
    
    model  = generate_model()
    device = torch.device("cpu")
    model.to(device)

    train_loader,val_loader, test_loader = generate_data()
    loss_module = torch.nn.BCELoss()
    optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr, betas = (0.9,0.99))
    
    train_model(model       = model,
                       device      = device,
                       optimizer   = optimizer,
                       loss_module = loss_module,
                       data_loader = train_loader,
                       num_epochs  = args.num_epochs)
    print("Done")