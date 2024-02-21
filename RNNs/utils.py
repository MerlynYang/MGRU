# write a class of data
# which will do transformation and de-transformation
# generate the input of the data
import numpy as np
import pandas as pd
import torch

# we use this class to facilitate the data generation of batch data
# although in our experiment the batch_size is 1
class RV_data():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.X, self.Y

class RV_dataset():
    def __init__(self, train_size, val_size, test_size, rolling_step, forecast_horizon):
        self.raw_data = pd.read_csv('data/realized_volatility/log_rv_table.csv', index_col=0)
        self.train_size = train_size # number of training samples
        self.val_size = val_size # number of validation samples
        self.test_size = test_size # number of testing samples
        self.sample_size = len(self.raw_data)
        self.rolling_step = rolling_step # update the rolling window (re-estimate steps)
        self.forecast_horizon = forecast_horizon
        
        self.normalization()
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
        return RV_data(self.norm_data[(start-self.forecast_horizon):(end-self.forecast_horizon), :], self.norm_data[(start):(end), :])

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
        self.norm_data = 2 * (self.raw_data - self.in_sample_min.min()) / (self.in_sample_max.max() - self.in_sample_min.min()) - 1

    def denormalization(self, forecasts):
        # forecast : [rolling step, stock number]
        return (forecasts + 1) * (self.in_sample_max.max() - self.in_sample_min.min()) / 2 + self.in_sample_min.min()
