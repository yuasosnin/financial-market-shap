import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import datetime, math, json


def columns_suffix(df, name):
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df.set_index('date', drop=True)
    df.columns = [f'{c}_{name}' for c in df.columns]
    df = df.reset_index()
    return df

def multi_merge(df_list, on=None, how='outer'):
    from functools import reduce
    return reduce(lambda df1, df2: pd.merge(df1, df2, on=on, how=how), df_list)

def robust_mean(x, p=0.05):
    low = x.quantile(p)
    high = x.quantile(1-p)
    return x[(x>low) & (x<high)].mean()

def robust_std(x, p=0.05):
    low = x.quantile(p)
    high = x.quantile(1-p)
    return x[(x>low) & (x<high)].std()


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
    def __init__(self, df_x, y, seq_length=10, steps=1, task='regression', norm_stats=None):
        self.seq_length = seq_length
        self.steps = steps
        self.task = task
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
        if norm_stats is not None:
            self.norm_stats = torch.tensor(norm_stats)
            self.data = (self.data - self.norm_stats[0]) / self.norm_stats[1]
        else: self.norm_stats = None

    def __len__(self):
        return self.data.shape[0] - self.seq_length - self.steps

    def __getitem__(self, idx): 
        x = self.data[idx:idx+self.seq_length, self.x_idx].float()
        if isinstance(self.y_idx, int):
            y = self.data[idx:idx+self.seq_length, [self.y_idx]].float()
        else:
            y = self.data[idx:idx+self.seq_length, self.y_idx].float()
        target = self.data[idx+self.seq_length:idx+self.seq_length+self.steps, self.y_idx].float()#.squeeze(0)
        if self.task == 'classification':
            target = (target > y[-1]).float()
        return (
            x, # x lags
            y, # y lags
            (target if isinstance(self.y_idx, int) else target.squeeze(0)) # target
        )

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
        
    def tensor_apply(self, x, fun):
        x_d = torch.zeros_like(x)
        for i, t in enumerate(x):
            x_d[i] = torch.from_numpy(fun(t.numpy()))
        return x_d

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

    
def take_vars(data, idx, var, add=''):
    take = [f'{v}_{i}' for i in idx for v in var]
    take = list(set(take)&set(data.columns))
    take = take + [c for c in data.columns if c.split('_')[0] in add]
    return data[take].copy()


def convert_to_lagged_df(dataset, colnames=None):
    x,y,target = dataset[0]
    num_lags = y.shape[0]
    num_predictors = x.shape[1]
    if colnames is None:
        colnames = [f'x_{i}' for i in range(num_predictors)]
    assert len(colnames) == num_predictors
    gen = (np.concatenate([target.numpy(), y.flatten().flip(0).numpy(), x.flip(0).flatten().numpy()]) for (x,y,target) in dataset)
    return pd.DataFrame(gen, columns=['y_target']+[f'y_lag{i+1}' for i in range(num_lags)]+[f'{colnames[j]}_lag{i+1}' for i in range(num_lags) for j in range(num_predictors)])

    
def plot_preds(y_pred, y_true, splits=None):
    fig = plt.figure(figsize=(24,6))
    plt.plot(y_true)
    plt.plot(y_pred)
    plt.legend(('True', 'Pred'))
    if splits is not None:
        for split in splits:
            plt.axvline(x=split, color='r', linestyle='--')
    return fig

def plot_cumsum(y_pred, y_true):
    cumsum = np.cumsum((y_true - y_pred)**2)
    cumsum_zero = np.cumsum((y_true)**2)
    fig = plt.figure()
    plt.plot(cumsum_zero)
    plt.plot(cumsum)
    plt.title('Squared Error Cumulative Sum')
    plt.legend(('Zero', 'Model'))
    return fig


def plot_loss(train_loss, valid_loss, same_axis=True, **kwargs):
    if same_axis == True:
        fig, ax1 = plt.subplots(**kwargs)
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('train loss')
        plot1 = ax1.plot(train_loss, label='Train')
        # ax1.legend()

        ax2 = ax1.twinx()
        ax2.set_ylabel('valid loss')
        plot2 = ax2.plot(valid_loss, label='Valid', color='orange')
        # ax2.legend()

        # added these three lines
        plots = plot1+plot2
        labs = [l.get_label() for l in plots]
        fig.legend(plots, labs, loc=1)
        plt.title('Loss')
        
        fig.tight_layout()
        return fig
    else:
        plt.plot(train_loss)
        plt.plot(valid_loss, color='orange')
        return None


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

    
def l_out(l_in, kernel_size, padding=0, dilation=1, stride=None):
    if stride is None:
        stride = kernel_size
    return int(((l_in + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1)


def number_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_params(model):
    return {k:v for k,v in model.__dict__.items() if k.split('_')[0] not in ('', 'training')}