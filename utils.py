import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from tqdm.notebook import tqdm


def calculate_normalization_statictics(df: pd.DataFrame, train_size=1, exclude=None):
    if isinstance(train_size, float):
        assert train_size > 0 and train_size <= 1
        train_size = round(len(df) * train_size)
    else:
        assert isinstance(train_size, int) and train_size <= len(df)-1
    exclude = exclude or []
    
    means = []
    stds = []
    for column in df.columns:
        if column in exclude:
            means.append(0)
            stds.append(1)
        else:
            means.append(df[column].mean())
            stds.append(df[column].std())
    return [means, stds]


def sequential_split(dataset: Dataset, splits):
    '''
    A function similar to torch.utils.data.random_split
    Splits dataset sequantially into len(splits)+1 parts
    `splits` can either be an iterable of shares of each part, summing up to 1,
    or directly indexes to split at, with splits[-1] == len(dataset)-1
    Returns a list of datasets of class Subset
    '''
    indexes = all(isinstance(i, int) for i in splits)
    shares = all((isinstance(i, float) and i > 0) for i in splits) and sum(splits) - 1 < 10e-6
    assert indexes or shares

    splits = [0] + splits
    if shares:
        splits = [round(len(dataset) * sum(splits[:i+1])) for i, _ in enumerate(splits)]

    return [Subset(dataset, range(i, j)) for i, j in zip(splits[:-1], splits[1:])]


class TickerDataset(Dataset):
    def __init__(self, df_x, y, seq_length=10, norm_stats=None):
        self.seq_length = seq_length
        self.data = torch.tensor(df_x.values).float()
        if isinstance(y, int):
            self.y_idx = y
        elif isinstance(y, str):
            self.y_idx = list(df_x.columns).index(y)
        self.x_idx = [i for i in range(self.data.shape[1]) if i != self.y_idx]
        if norm_stats is not None:
            self.norm_stats = torch.tensor(norm_stats)
            self.data = (self.data - self.norm_stats[0]) / self.norm_stats[1]
        else: self.norm_stats = None

    def __len__(self):
        return self.data.shape[0] - self.seq_length - 1

    def __getitem__(self, idx): 
        return (
            self.data[idx:idx+self.seq_length, :].float(), # x lags
            self.data[idx:idx+self.seq_length, [self.y_idx]].float(), # y lags
            self.data[idx+self.seq_length, [self.y_idx]].float() # y current
        )

