import torch
import numpy as np
from time import time
import pandas as pd
import argparse
import random

import models
from arfima_smi import arfima

def simulation(d:float, T:int, ar:list, ma:list):
    replication = 500
    d_hat_list = []
    
    for r in range(replication):
        # set the seed
        torch.manual_seed(r)
        np.random.seed(r)
        random.seed(r)
        
        random_number = np.random.uniform(-1, 1, 1)[0]
        if d + 0.05 * random_number < 0:
            init_mem = [d - 0.05 * random_number]
        else:
            init_mem = [d + 0.05 * random_number]
        
        # gerenate the time series
        series = arfima(ar, d, ma, T, warmup=T)
        series = 2 * (series - np.min(series)) / (np.max(series) - np.min(series)) - 1
        X = torch.from_numpy(series[:-1].reshape(1, -1, 1)).float()
        Y = torch.from_numpy(series[1:].reshape(1, -1, 1)).float()
        train_X = X[:, :int(0.8 * T), :]
        train_Y = Y[:, :int(0.8 * T), :]
        val_X = X[:, int(0.8 * T):, :]
        val_Y = Y[:, int(0.8 * T):, :]
        # set training 
        model = models.MRNNFModel('MGRUF', 1, 1, 1, 100, 'stack', 'cpu', 0, init_mem)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        start_time = time()
        stuck = 0
        min_val_loss = np.inf
        for epoch in range(500):
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
                # d_hat of the best model
                d_hat = 0.5 * torch.nn.functional.sigmoid(model.mem_para.detach())
            else:
                stuck += 1
            if stuck > 50:
                break
        print(f'loss = {loss.item():.3f}, init_d = {init_mem[0]:.3f}, d_hat = {d_hat.numpy()[0]:.3f}, time = {(time() - start_time):.1f}')
        d_hat_list.append(d_hat.numpy())
    print(f'd = {d}, T = {T}, AR = {ar}, MA = {ma}, d_hat_mean = {np.mean(d_hat_list)},  bias = {d - np.mean(d_hat_list)}, std_d_hat = {np.std(d_hat_list)}')
    d_df = pd.DataFrame()
    d_df[f'd={d}'] = d_hat_list
    d_df.to_csv(f'results/simulations/T_{T}_ar_{ar[0]}_ma_{ma[0]}_d_{d}.csv')
    
    return d_hat_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=float, default=0.1)
    parser.add_argument('--T', type=int, default=500)

    args = parser.parse_args()

    # specify the AR and MA parameters
    simulation(args.d, args.T, [0.7], [0.5])
