# This file is used to simulate an ARMA process and compare the performance of GRU and MGRU on it.
import numpy as np
import statsmodels.api as sm
import torch
import models
import pandas as pd
from time import time
import argparse

def simulation(T:int, ar:list, ma:list):
    np.random.seed(1234)
    torch.manual_seed(1234)
    ar = np.array(ar)
    ma = np.array(ma)
    ar_param = np.r_[1, -ar]
    ma_param = np.r_[1, ma]
    data = sm.tsa.arma_generate_sample(ar_param, ma_param, T)
    return data

def train(model_name, data):
    start_time = time()
    if model_name == 'GRU':
        model = models.RNNModel('GRU', 1, 1, 1, 'stack', 'cpu', 0)
    elif model_name == 'MGRUF':
        model = models.MRNNFModel('MGRUF', 1, 1, 1, 50, 'stack', 'cpu', 0)
    else:
        raise ValueError('Other rnn_type not supported yet')
    
    data_min = np.min(data)
    data_max = np.max(data)
    data_norm = 2 * (data - data_min) / (data_max - data_min) - 1
    
    X = torch.from_numpy(data_norm[:-1].reshape(1, -1, 1)).float()
    Y = torch.from_numpy(data_norm[1:].reshape(1, -1, 1)).float()
    
    train_X = X[:, :train_size, :]
    train_Y = Y[:, :train_size, :]
    val_X = X[:, train_size:(train_size+val_size), :]
    val_Y = Y[:, train_size:(train_size+val_size), :]
    test_X = X[:, (train_size+val_size):, :]
    test_Y = Y[:, (train_size+val_size):, :]
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    stuck = 0
    min_val_loss = np.inf
    
    for epoch in range(max_epoch):
        model.train()
        optimizer.zero_grad()
        output, _ = model(train_X)
        loss = criterion(output, train_Y)
        loss.backward()
        optimizer.step()
        
        model.eval()
        output, _ = model(val_X)
        val_loss = criterion(output, val_Y)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            stuck = 0
        else:
            stuck += 1
        if stuck >= 10:
            break
        
    model.eval()
    output, _ = model(test_X)
    test_loss = criterion(output, test_Y)
    
    recover_output = (output + 1) * (data_max - data_min) / 2 + data_min
    diff = recover_output - test_Y
    RMSE = torch.sqrt(torch.mean(diff ** 2)).item()
    MAE = torch.mean(torch.abs(diff)).item()
    
    print(f'time = {(time() - start_time):.1f}, train_loss = {loss.item():.3f}, val_loss = {min_val_loss.item():.3f}, test_loss = {test_loss.item():.3f}')
    print(f'RMSE = {RMSE:.3f}, MAE = {MAE:.3f}')
    
    return RMSE, MAE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arma', type=str, default='arma')
    parser.add_argument('--T', type=int, default=2000)
    parser.add_argument('--model', type=str, default='MGRUF')

    args = parser.parse_args()
    
    replication = 100

    if args.arma == 'arma':
        ar = [0.2]
        ma = [0.2]
    elif args.arma == 'ar':
        ar = [0.6]
        ma = [0]
    else:
        raise ValueError('Other ARMA type not supported yet')
    
    T = args.T
    rnn = args.model
    
    train_size = int(T * 0.75)
    val_size = int(T * 0.15)
    test_size = int(T * 0.15)
    max_epoch = 500
    
    print(f'rnn : {rnn}, ar : {ar[0]}, ma : {ma[0]}')
    
    result_df = pd.DataFrame(columns=[f'{rnn}_RMSE', f'{rnn}_MAE'])
    
    data = simulation(T, ar, ma)

    for r in range(replication):
        np.random.seed(r)
        torch.manual_seed(r)
        
        RMSE, MAE = train(rnn, data)
        result_df.loc[r, f'{rnn}_RMSE'] = RMSE
        result_df.loc[r, f'{rnn}_MAE'] = MAE

    result_df.to_csv(f'results/simulations/{args.model}_{args.T}_{args.arma}.csv')