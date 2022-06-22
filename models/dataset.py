import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
import pytorch_lightning as pl

from .utils import *


def calculate_norm_stats(df: pd.DataFrame, train_size=1, exclude=[], only_std=[], robust=False):
    if isinstance(train_size, float):
        assert train_size > 0 and train_size <= 1
        train_size = round(len(df) * train_size)
    else:
        assert isinstance(train_size, int) and train_size <= len(df)-1
    exclude = exclude or []

    if robust:
        mean = robust_mean
        std = robust_std
    else:
        mean = lambda x: x.mean()
        mean = lambda x: x.std()

    means = []
    stds = []
    for column in df.columns:
        if column.split('_')[0] in exclude:
            means.append(0)
            stds.append(1)
        elif column.split('_')[0] in only_std:
            means.append(0)
            stds.append(std(df.iloc[:train_size][column]))
        else:
            means.append(np.mean(df.iloc[:train_size][column]))
            stds.append(np.std(df.iloc[:train_size][column]))
    return [means, stds]


class TickerDataset(Dataset):
    def __init__(self, df_x, y, seq_length=10, norm_stats=None):
        self.seq_length = seq_length
        if norm_stats is not None:
            self.norm_stats = norm_stats
            data = (df_x.copy() - self.norm_stats[0]) / self.norm_stats[1]
            self.data = torch.tensor(data.values).float()
        else:
            self.norm_stats = None
            self.data = torch.tensor(df_x.values).float()
        if isinstance(y, list):
            if all([isinstance(i, int) for i in y]):
                self.y_idx = y
            elif all([isinstance(i, str) for i in y]):
                self.y_idx = [list(df_x.columns).index(i) for i in y]
        elif isinstance(y, int):
            self.y_idx = y
        elif isinstance(y, str):
            self.y_idx = list(df_x.columns).index(y)
        self.x_idx = [i for i in range(self.data.shape[1]) if (i not in (self.y_idx if isinstance(self.y_idx, list) else [self.y_idx]))]

    def __len__(self):
        return self.data.shape[0] - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length, self.x_idx].float()
        if isinstance(self.y_idx, int):
            y = self.data[idx:idx+self.seq_length, [self.y_idx]].float()
        else:
            y = self.data[idx:idx+self.seq_length, self.y_idx].float()
        target = self.data[idx+self.seq_length, [self.y_idx]].float()#.squeeze(0)
        return (
            x,
            y,
            target
        )


class MyDataModule(pl.LightningDataModule):
    def __init__(self, data, target, seq_length=5, batch_size=32, num_workers=1, splits=[0.8, 0.1, 0.1], norm_stats=None):
        super().__init__()
        self.data = data
        self.target = target
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splits = splits
        self.norm_stats = norm_stats

    def setup(self, stage=None):
        self.norm_stats = calculate_norm_stats(self.data, train_size=self.splits[0], exclude=['time','dayofweek'], robust=True)
        # self.denoise = lambda x: denoise_wavelet(x, method='BayesShrink', mode='soft', wavelet_levels=3, wavelet='Haar')
        self.full_dataset = TickerDataset(self.data, y=self.target, seq_length=self.seq_length, norm_stats=self.norm_stats)
        self.train_dataset, self.val_dataset, self.test_dataset = sequential_split(self.full_dataset, splits=self.splits)
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.full_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)
