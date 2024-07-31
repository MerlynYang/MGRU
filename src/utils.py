# write a class of data
# which will do transformation and de-transformation
# generate the input of the data

import numpy as np
import pandas as pd
import torch

class RV_data():
    def __init__(self, arch_type, min_window, X, Y):
        self.arch_type = arch_type
        self.min_window = min_window
        self.X = X
        self.Y = Y
        
    def __len__(self):
        if self.arch_type == 'stack':
            return 1
        elif self.arch_type == 'slide':
            assert len(self.X) - self.min_window == len(self.Y) or len(self.X) - self.min_window < 0
            return len(self.Y)
        else:
            raise ValueError('Dataset type not implemented yet')
    
    def __getitem__(self, idx):
        if self.arch_type == 'stack':
            return self.X, self.Y
        elif self.arch_type == 'slide':
            return self.X[idx:(idx+self.min_window+1), :], self.Y[idx, :]

class RV_dataset():
    def __init__(self, train_size, val_size, test_size, min_window, rolling_step, forecast_horizon, norm_method, arch_type, test_func=False):
        self.raw_data = pd.read_csv('data/realized_volatility/log_rv_table.csv', index_col=0)
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.sample_size = len(self.raw_data)
        if self.train_size + self.val_size + self.test_size > len(self.raw_data):
            raise ValueError('input splits are larger than the data size')
        self.rolling_step = rolling_step # update the rolling window
        self.forecast_horizon = forecast_horizon
        self.norm_method = norm_method # normalization method
        self.arch_type = arch_type # stack or slide
        self.min_window = min_window
        
        self.normalization()
        if test_func:
            custom_data = np.array([[1 * j for i in range(len(self.norm_data.columns))] for j in range(len(self.norm_data.index))], dtype=np.int32)
            self.norm_data = pd.DataFrame(custom_data, index=self.norm_data.index, columns=self.norm_data.columns)
            self.norm_data = torch.from_numpy(self.norm_data.values).float()
        else:
            self.norm_data = torch.from_numpy(self.norm_data.values).float()
        
        self.this_step = 0
        self.cal_bound_for_target()
    
    def cal_bound_for_target(self):
        if self.this_step == 0:
            self.train_start = self.forecast_horizon
            self.train_end = self.train_size
            self.val_end = self.val_size + self.train_size
            self.test_end = self.train_size + self.val_size + self.rolling_step
        else:
            self.train_end += self.rolling_step
            self.train_start = self.train_end - self.train_size
            self.val_end += self.rolling_step
            if self.this_step == self.rolling_step:
                self.test_end += self.rolling_step
            else:
                self.test_end = self.sample_size
    
    def get_data(self, data_type):
        if data_type == 'train':
            start = self.train_start
            end = self.train_end
        elif data_type == 'val':
            start = self.train_end
            end = self.val_end
        elif data_type == 'test':
            start = self.val_end
            end = self.test_end
        if self.arch_type == 'stack':
            return RV_data(self.arch_type, self.min_window, self.norm_data[(start-self.forecast_horizon):(end-self.forecast_horizon), :], self.norm_data[(start):(end), :])
        elif self.arch_type == 'slide':
            if start < self.min_window + self.forecast_horizon:
                start = self.min_window + self.forecast_horizon
            return RV_data(self.arch_type, self.min_window, self.norm_data[(start-self.min_window-self.forecast_horizon):(end-self.forecast_horizon)], self.norm_data[(start):(end), :])

    def rolling(self):
        dist_to_end = self.sample_size - self.test_end
        if dist_to_end <= 0:
            return False
        self.this_step = self.rolling_step if dist_to_end >= self.rolling_step else dist_to_end
        self.cal_bound_for_target()
        return True
    
    def normalization(self):
        self.in_sample_min = np.array(self.raw_data.iloc[:(self.train_size + self.val_size), :].min()).reshape(1, -1)
        self.in_sample_max = np.array(self.raw_data.iloc[:(self.train_size + self.val_size), :].max()).reshape(1, -1)
        if self.norm_method == 'over_all':
            self.norm_data = 2 * (self.raw_data - self.in_sample_min.min()) / (self.in_sample_max.max() - self.in_sample_min.min()) - 1
        elif self.norm_method == 'individual':
            self.norm_data = 2 * (self.raw_data - self.in_sample_min) / (self.in_sample_max - self.in_sample_min) - 1
            self.norm_mean = np.array(self.norm_data.iloc[:(self.train_size + self.val_size), :].mean()).reshape(1, -1)
            self.norm_data = self.norm_data - self.norm_mean
        else:
            NotImplementedError('Other normalization methods not implemented yet')
        return

    def denormalization(self, forecasts):
        # forecast : [rolling step, stock number]
        if self.norm_method == 'over_all':
            return (forecasts + 1) * (self.in_sample_max.max() - self.in_sample_min.min()) / 2 + self.in_sample_min.min()
        elif self.norm_method == 'individual':
            return (forecasts + self.norm_mean + 1) * (self.in_sample_max - self.in_sample_min) / 2 + self.in_sample_min
        else:
            NotImplementedError('Other normalization methods not implemented yet')
        return None
