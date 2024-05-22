import numpy as np
import os
import pandas as pd
import torch
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.timeseries_data import Dataset_ETT_hour, Dataset_Custom

station_idx_WeatherTiny = {0: 79, 1: 114, 2: 209, 3: 259, 4: 532, 5: 546, 6: 850, 7: 852, 8: 947, 9: 1201, 10: 1272, 11: 1328, 12: 1527, 13: 1691, 14: 1786}
# 'PRECTOTPS	QV2M	T2M	T2MDEW	T2MWET	T2M_MAX	T2M_MIN	T2M_RANGE	TS	WS10M	WS10M_MAX	WS10M_MIN	WS10M_RANGE	WS50M	WS50M_MAX	WS50M_MIN	WS50M_RANGE'

# For USCAIRCN dataset
# city_name = ['Akesu', 'Albuquerque', 'Atlanta', 'Beersheba', 'Boston', 'Changsha', 'Charlotte', 'Chicago', 'Dallas', 'Denver', 
#         'Detroit', 'Eilat', 'Fuzhou', 'Guangzhou', 'Guiyang', 'Haerbin', 'Haidian', 'Haifa', 'Houston', 'Huhehaote', 
#         'Indianapolis', 'Jacksonville', 'Jerusalem', 'Kansas City', 'Kunming', 'Las Vegas', 'Linzhi', 'Los Angeles', 
#         'Miami', 'Minneapolis', 'Montreal', 'Nahariyya', 'Nashville', 'New York', 'Philadelphia', 'Phoenix', 'Pittsburgh', 
#         'Portland', 'Pudong', 'Saint Louis', 'San Antonio', 'San Diego', 'San Francisco', 'Seattle', 'Shenyang', 'Tel Aviv District', 
#         'Toronto', 'Vancouver', 'Wuhan', 'Xindu', 'Yinchuan']
city_name = ['Albuquerque', 'Atlanta', 'Beersheba', 'Boston', 'Charlotte', 'Chicago', 'Dallas', 'Denver', 
        'Detroit', 'Eilat', 'Haifa', 'Houston',
        'Indianapolis', 'Jacksonville', 'Jerusalem', 'Kansas City', 'Las Vegas', 'Los Angeles', 
        'Miami', 'Minneapolis', 'Montreal', 'Nahariyya', 'Nashville', 'New York', 'Philadelphia', 'Phoenix', 'Pittsburgh', 
        'Portland', 'Saint Louis', 'San Antonio', 'San Diego', 'San Francisco', 'Seattle', 'Tel Aviv District', 
        'Toronto', 'Vancouver']

def read_client_data(args, idx, is_train=True):
    data_dir = os.path.join('../dataset', args.dataset)
    if 'Weather' in args.dataset:
        station_idx = station_idx_WeatherTiny
    elif args.dataset == 'DroughtED':
        file_count = len([file for file in os.listdir(data_dir) if file.endswith('.csv')])
        station_idx_DroughtED = {}
        for i in range(file_count):
            station_idx_DroughtED[i] = f'{i}'
        station_idx = station_idx_DroughtED
    elif args.dataset == 'USCAIRCN':
        station_idx = city_name
        
    if is_train:
        train_data = Dataset_Custom(root_path=data_dir, flag='train', size=[args.seq_len, args.label_len, args.pred_len], 
                                    features=args.features, data_path='{}.csv'.format(station_idx[idx]), target=args.target, 
                                    timeenc=1, freq=args.freq)
        return train_data
    else:
        test_data = Dataset_Custom(root_path=data_dir, flag='test', size=[args.seq_len, args.label_len, args.pred_len], 
                                   features=args.features, data_path='{}.csv'.format(station_idx[idx]), target=args.target, 
                                   timeenc=1, freq=args.freq)
        return test_data

import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

