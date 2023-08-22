import torch

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

all_data = pd.read_csv('/**/data/all_data.csv', index_col=['rgi_id', 'period'])

# setting up test/train split
g_idx = np.unique(all_data.index.get_level_values(0).values)

g_train, g_test = train_test_split(g_idx, train_size=0.2,test_size=0.8)

train_df = all_data.loc[g_train]
test_df = all_data.loc[g_test]

testing_df = test_df.to_csv('/**/data/df_test.csv')

g_idx = np.unique(train_df.index.get_level_values(0).values)

train_dataset, validation_dataset = train_test_split(g_idx, train_size=0.85, test_size=0.15)

train_df = all_data.loc[train_dataset]
val_df = all_data.loc[validation_dataset]

## ---- data set and data loader code for pytorch 

torch.manual_seed(132)

class Sequencer(Dataset):
    def __init__(self, data, seq_length):
        self.seq_length = seq_length

        features_to_drop = ['dmdtda', 'err_dmdtda', 'target_id']

        data_hold = data.drop(features_to_drop, axis=1)

        scaler = StandardScaler()
        scaler.fit(data_hold)

        scaled_X = scaler.transform(data_hold)

        self.X = torch.tensor(scaled_X).float() 
        self.y = torch.tensor(data[['dmdtda']].values).float()

    def __len__(self): 
        return self.data.shape[0]
    
    def __getitem__(self, i): 
        if i >= self.seq_length - 1:
            i0 = i - self.seq_length + 1
            next_X = self.X[i0:(i+1), :]
        else:
            padding = self.data[0].repeat(self.sequence_length - i - 1, 1) 
            next_X = self.X[0:(i+1), :]
            next_X = torch.cat((padding, next_X), 0)

        return next_X, self.y[i]
    

seq_length = 5
    
train_data = Sequencer(train_df, seq_length)
train_load = DataLoader(train_data, batch_size=5, shuffle=True, num_workers=1)

val_data = Sequencer(val_df, seq_length)
val_load = DataLoader(val_data, batch_size=5, shuffle=True, num_workers=1)

test_data = Sequencer(test_df, seq_length)
test_load = DataLoader(test_data, batch_size=5, shuffle=True, num_workers=1)

##-------- model 

class RegModLSTM(nn.Module):
    def __init__(self, input_feat, h):
        super().__init__()
        self.input_feat = input_feat
        self.h = h
        self.n_layers = 7

        self.lstm = nn.LSTM(input_size=input_feat, hidden_size=h, num_layers = self.n_layers, dropout = 0.1, 
                            batch_first=True)

        self.lin1 = nn.Linear(in_features=h, out_features=1)
    
    def forward(self, data):
        batch_size = data.shape[0]
        
        h_p = torch.zeros(self.n_layers, batch_size, self.h).requires_grad_()
        c_p = torch.zeros(self.n_layers, batch_size, self.h).requires_grad_()

        _, (h_n, _) = self.lstm(data, (h_p, c_p))

        return self.lin1(h_n[0]).flatten()
    
##-------- implementation 

mod = RegModLSTM(input_feat=34, h=200)
loss_func = nn.MSELoss()
opt = torch.optim.Adam(mod.parameters(), 0.0005)

## standard functiions for PyTorch training and testing models 

def training_mod(data_loader, model, loss_func, optim):     
    batch_size = len(data_loader)
    t_loss = 0
    model.train()

    for X,y in data_loader:

        out = model(X)
        loss = loss_func(out, y)

        # clear gradients for this training step
        optim.zero_grad()

        # backpropagation, compute gradients 
        loss.backward()

        # apply gradients 
        optim.step()

        t_loss += loss.item() 

    avg_loss = t_loss/batch_size

    print(f"training loss: {avg_loss}")

def testing_mod(data_loader, model, loss_func):

    batch_size = len(data_loader)
    t_loss = 0

    model.eval()

    with torch.no_grad():
        for X,y in data_loader:

            out = model(X)
            t_loss += loss_func(out, y).item() 

    avg_loss = t_loss/batch_size

    print(f"training loss: {avg_loss}")


    










    







