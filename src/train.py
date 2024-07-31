# class experiment
# given hyper-parameter and data
# return the best model (validation loss) corresponding to the data

# class experiment_manager:
# given a specification of experiment
# get the best model (validation loss)
# get the forecasts
# comparing different specifications of experiment
# get the final forecasts
import torch
import numpy as np
from time import time
from utils import RV_dataset
import models
from copy import deepcopy
from torch.utils.data import DataLoader
import random
import pandas as pd
import argparse
import math

class experiment():
    # given a specification of hyper-parameters and data
    # return test loss and forecasts of the best model (according to validation loss)
    def __init__(self, train_loader, val_loader, test_loader, **args):
        # set seed for reproducibility
        torch.manual_seed(args['seed'])
        np.random.seed(args['seed'])
        random.seed(args['seed'])
        if args['device'] == 'cuda':
            torch.cuda.manual_seed(args['seed'])

        self.args = args
        self.device = self.args['device']
        # set model
        if self.args['rnn_type'] == 'RNN' or self.args['rnn_type'] == 'GRU' or self.args['rnn_type'] == 'LSTM':
            self.model = models.RNNModel(args['rnn_type'], args['input_size'], args['hidden_size'], args['output_size'], args['arch_type'], args['device'], args['dropout']).to(self.device)
        elif self.args['rnn_type'] == 'MGRUF' or self.args['rnn_type'] == 'MLSTMF':
            self.model = models.MRNNFModel(args['rnn_type'], args['input_size'], args['hidden_size'], args['output_size'], args['lag_k'], args['arch_type'], args['device'], args['dropout']).to(self.device)
        elif self.args['rnn_type'] == 'MGRU' or self.args['rnn_type'] == 'MLSTM':
            pass

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args['lr'])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def train_epoch(self):
        hidden = None
        total_train_loss, total_val_loss = 0, 0
        train_batchs, val_batchs = 0, 0
        
        # for stack : X [batch_size = 1, seq_len, input_size], Y [batch_size = 1, seq_len, output_size] and need to save hidden states
        # for slide : X [batch_size, min_window, input_size], Y[batch_size, input_size] and hidden state for each batch is re-initialized
        
        self.model.train()
        for X, Y in self.train_loader:
            self.optimizer.zero_grad()
            if self.args['arch_type'] == 'stack': # this can be change to weighted loss
                # keep the hidden information
                train_output, hidden = self.model(X.to(self.device), hidden)
                # everytime the hidden is re-initialized to None
                train_loss = self.criterion(train_output, Y.to(self.device))
            elif self.args['arch_type'] == 'slide':
                # no historical hidden
                train_output, _ = self.model(X.to(self.device))
                train_loss = self.criterion(train_output, Y.to(self.device))
            else:
                raise ValueError('Dataset type not implemented yet')
            total_train_loss += train_loss.cpu().item() * X.shape[0]
            train_batchs += X.shape[0]
            train_loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            self.model.eval()
            for X, Y in self.val_loader:
                X = X.to(self.device)
                Y = Y.to(self.device)
                if self.args['arch_type'] == 'slide':
                    val_output, hidden = self.model(X.to(self.device))
                elif self.args['arch_type'] == 'stack':
                    assert hidden is not None
                    val_output, hidden = self.model(X.to(self.device), hidden)
                else:
                    raise ValueError('Dataset type not implemented yet')
                val_loss = self.criterion(val_output, Y.to(self.device))
                total_val_loss += val_loss.cpu().item() * X.shape[0]
                val_batchs += X.shape[0]
        return total_train_loss / train_batchs, total_val_loss / val_batchs

    def train_model(self):
        self.min_val_loss = np.inf
        self.best_model = None
        stuck = 0
        patience = self.args['patience']
        epoch = 1
        # ! can change
        max_epoch = 1000
        train_loss, val_loss = 0, 0
        
        start_time = time()
        last_time = time()
        
        while (stuck < patience and epoch < max_epoch):
            train_loss, val_loss = self.train_epoch()
            if val_loss < self.min_val_loss:
                self.min_val_loss = val_loss
                self.best_model = deepcopy(self.model)
                stuck = 0
            elif math.isnan(train_loss) or math.isnan(val_loss):
                print('!' * 18 + ' encounter NAN ' + '!' * 18)
                break
            else:
                stuck += 1
            if epoch % 1000 == 0:
                print(f'epoch {epoch:4d} : train loss {train_loss:.6f}, val loss {val_loss:.6f}, time {time() - last_time:.2f}')
                last_time = time()
            epoch += 1
        test_loss, _ = self.evaluate()
        
        # print(f'epoch {epoch:4d} : train last {train_loss:.6f}, val mini {self.min_val_loss:.6f}, time {time() - start_time:.2f}')
        print(f'epoch {epoch:4d} : test  loss {test_loss:.6f}, time {time() - start_time:.2f}')
        
        return self.min_val_loss

    def evaluate(self):
        if self.best_model is None:
            return 999999, 999999
        self.best_model = self.best_model.to(self.device)
        self.best_model.eval()
        hidden = None
        total_test_loss = 0
        total_batchs = 0
        test_output_list = []
        
        for X, Y in self.test_loader:
            if self.args['arch_type'] == 'stack':
                # warm up, for stack to update the hidden state
                for train_X, _ in self.train_loader:
                    _, hidden = self.best_model(train_X.to(self.device), hidden)
                for val_X, _ in self.val_loader:
                    _, hidden = self.best_model(val_X.to(self.device), hidden)
                assert hidden is not None
                test_output, hidden = self.best_model(X.to(self.device), hidden)
            elif self.args['arch_type'] == 'slide':
                test_output, hidden = self.best_model(X.to(self.device))
            else:
                raise ValueError('Dataset type not implemented yet')
            test_loss = self.criterion(test_output, Y.to(self.device))
            total_test_loss += test_loss.cpu().item() * X.shape[0]
            total_batchs += X.shape[0]
            if self.args['arch_type'] == 'stack':
                test_output_list.append(test_output.cpu().detach().numpy().squeeze(0))
            elif self.args['arch_type'] == 'slide':
                test_output_list.append(test_output.cpu().detach().numpy())
            else:
                raise ValueError('Dataset type not implemented yet')
        
        self.best_model = self.best_model.cpu()
        # test loss
        return total_test_loss / total_batchs, np.concatenate(test_output_list, axis=0)

class experiment_manager():
    # given a list of specifications
    # run the experiments for each specification
    # save the data
    def __init__(self, rnn_type, forecast_horizon, **args_list):
        self.args_list = args_list
        self.default_config = {
            # 'rnn_type': 'MGRUF', # save name
            # trading days per year, [245, 244, 244, 244, 243, 244, 243, 243, 242, 242]
            'input_size': 152, # fixed
            'output_size': 152, # fixed
            'train_size': 1950, # fixed
            'val_size': 242, # fixed
            'test_size': 242, # fixed
            'rolling_step': 22, # fixed
            'norm_method': 'over_all', # fixed
            'patience': 100, # fixed
            'device': 'cpu', # fixed # ! this can be change to cuda
            'seed': 42, # fixed
            'arch_type': 'stack', # save name (fixed)
            'min_window': 666, # hyper-for slide (fixed for stack)
            'batch_size': 64, # hyper-parameter (fixed for stack)
            
            # the pre-experiment result shows that :
            # learning rate with 0.001 is better
            # dropout with 0 is better
            # hidden size with 512 is better
            
            'dropout': 0, # hyper-parameter [0, 0.1, 0.2, 0.3]
            'hidden_size': 256, # hyper-parameter [64, 128, 256, 512, 768]
            'lr': 0.001, # hyper-parameter [0.005, 0.001, 0.0005, 0.0001, 0.00001]
            'lag_k': 10 #hyper-parameter for long-memory model [20, 30, 40, 50]
        }
        self.default_config['rnn_type'] = rnn_type
        self.default_config['forecast_horizon'] = forecast_horizon
        self.dataset = RV_dataset(self.default_config['train_size'], self.default_config['val_size'], self.default_config['test_size'], self.default_config['min_window'], self.default_config['rolling_step'], self.default_config['forecast_horizon'], self.default_config['norm_method'], self.default_config['arch_type'], False)

    # in a given window, given setting
    # return the validation loss and forecasts
    def get_forecast_given_setting(self, **args):
        # set hyper-parameters
        output_string = 'setting : '
        for name, value in args.items():
            self.default_config[name] = value
            output_string += f'{name}:{value}, '
        print(output_string[:-2])
        
        train_loader = DataLoader(self.dataset.get_data('train'), batch_size=self.default_config['batch_size'], shuffle=False)
        val_loader = DataLoader(self.dataset.get_data('val'), batch_size=self.default_config['batch_size'], shuffle=False)
        test_loader = DataLoader(self.dataset.get_data('test'), batch_size=self.default_config['batch_size'], shuffle=False)
        
        exp = experiment(train_loader, val_loader, test_loader, **self.default_config)
        validation_loss = exp.train_model()
        test_loss, forecasts = exp.evaluate()
        denormed_forecast = self.dataset.denormalization(forecasts) # remove the batch dimension
        
        return validation_loss, test_loss, denormed_forecast

    def get_forecast_whole_seq(self):
        all_forecasts = []
        rolling_time = 1
        
        while True:
            print(f'================   start training {rolling_time:3d}-th window   ================')
            min_val_loss = np.inf
            best_test_loss = 0
            best_forecast = None
            best_config = None
            # for all the args combination (accroding to the args list)
            # get the forecasts of this combination
            for drop_value in self.args_list['dropout']:
                for hidden_value in self.args_list['hidden_size']:
                    for lr_value in self.args_list['lr']:
                        for lag_value in self.args_list['lag_k']:
                            args = {
                                'dropout': drop_value,
                                'hidden_size': hidden_value,
                                'lr': lr_value,
                                'lag_k': lag_value
                            }
                            validation_loss, test_loss, denormed_forecast = self.get_forecast_given_setting(**args)
                            print('*' * 66)
                            if validation_loss < min_val_loss:
                                min_val_loss = validation_loss
                                best_forecast = denormed_forecast
                                best_config = args
                                best_test_loss = test_loss
            print(f'in this window, the best config is : {best_config}')
            print(f'with validation loss be {min_val_loss:.6f}, test loss be {best_test_loss}')
            print('=' * 66)
            print('\n')
            all_forecasts.append(best_forecast)
            if not self.dataset.rolling():
                break
            else:
                rolling_time += 1
        all_forecasts = np.concatenate(all_forecasts, axis=0)
        index_len, forecast_len = all_forecasts.shape
        # for multi-step forecast
        # the **model output** is use all the test data as target
        # but our specification is at the end of validation data to forecast
        # so the **true target** should be [forecast_horizon - 1:]
        
        df_forecasts = pd.DataFrame(all_forecasts, columns=self.dataset.raw_data.columns, index=self.dataset.raw_data.index[-index_len:]).iloc[(self.default_config['forecast_horizon'] - 1):, :]
        df_forecasts.to_csv(f"results/forecasts/{self.default_config['rnn_type']}_h_{self.default_config['forecast_horizon']}.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--forecast_horizon', type=int, default=1)
    parser.add_argument('--rnn_type', type=str, default='GRU')
    args = parser.parse_args()
    
    dropout_list = [0.0]
    lr_list = [0.001]
    hidden_list = [128, 256, 384, 512, 768]
    lag_k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # set possible experiment manager
    hyper_param_list = {
        'dropout': dropout_list,
        'hidden_size': hidden_list,
        'lr': lr_list,
        'lag_k': lag_k_list
    }
    
    exp_manager = experiment_manager(args.rnn_type, args.forecast_horizon, **hyper_param_list)
    exp_manager.get_forecast_whole_seq()
