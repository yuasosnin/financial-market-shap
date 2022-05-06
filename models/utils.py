import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import datetime, math, json


def _load_df(name, date, path=None):
    if path is not None:
        path = path.rstrip('/') + '/'
    else:
        path = ''
    data = pd.read_csv(f'{path}{name}_{date}.csv', sep=',', parse_dates=[['<DATE>', '<TIME>']])
    data.columns = map(lambda x: x.replace('<', '').replace('>', '').lower(), data.columns)
    if name == 'USD000000TOD':
        name = 'USD'
    if name == 'EUR_RUB__TOD':
        name = 'EUR'
    if name == 'ICE.BRN':
        name = 'BRENT'
    if name == 'SANDP-500':
        name = 'SNP'
    if name == 'comex.GC':
        name = 'GOLD'
    if name == 'FUTSEE-100':
        name = 'FTSE'
    for i in ['ticker', 'per']:
        try:
            data = data.drop(i, axis=1)
        except KeyError:
            pass
    data.columns = [column+('_'+name.lower())*bool(column not in ['date_time', 'date', 'time']) for column in data.columns]
    data = data.rename({'date_time': 'date', f'vol_{name.lower()}': f'volume_{name.lower()}'}, axis=1)
    #data = data.set_index('date')
    return data

def load_data(names, dates, path=''):
    from functools import reduce
    df_list = [pd.concat([_load_df(n, d, path) for d in dates]) for n in names]
    return reduce(lambda df1, df2: pd.merge(df1, df2, how='inner'), df_list)

def load_currency(name, path='data/'):
    data = pd.read_csv(f'{path}/{name}_RUB.csv', parse_dates=[0], dayfirst=True, decimal=',').iloc[::-1]
    name = name.lower()
    data.columns = ['date', 'close', 'open', 'high', 'low', 'change']
    data = data[['date', 'open', 'close', 'high', 'low', 'change']]
    data = data.drop(['change'], axis=1)
    data = data.set_index('date', drop=True)
    data['return'] = data['close'].pct_change()
    data['spread'] = (data['high'] - data['low']) / data['low']
    data = data[1:]
    data.columns = [f'{c}_{name.lower()}' for c in data.columns]
    return data

def load_yield(path='data/'):
    from functools import reduce
    
    dfs = []
    for i in [2,5,10,30]:
        data = pd.read_csv(f'{path}/DGS{i}.csv')
        data.columns = ['date', f'yield_{i}']
        data = data.loc[data[f'yield_{i}'] != '.', :]
        # data[f'yield_{i}'] = data[f'yield_{i}'].apply(lambda x: x.replace('.','0.'))
        data[f'yield_{i}'] = data[f'yield_{i}'].astype(float)
        dfs.append(data)
    
    data = reduce(lambda left, right: pd.merge(left, right, on='date', how='inner'), dfs)
    data = data.set_index('date', drop=True)
    data.index = pd.DatetimeIndex(data.index)
    return data


def prepare_data(data, idx):
    for i in idx:
        data[f'return_{i}'] = (data[f'close_{i}'] - data.shift()[f'close_{i}']) / data.shift()[f'close_{i}']
        data[f'log_return_{i}'] = np.log(data[f'close_{i}'] / data.shift()[f'close_{i}'])

    
    data['time_norm'] = 2 * math.pi * (data.date.dt.hour * 60 + data.date.dt.minute) / (24*60)
    data['time_cos'] = np.cos(data['time_norm'])
    data['time_sin'] = np.sin(data['time_norm'])
    data = data.drop('time_norm', axis=1)
    
    data['dayofweek_norm'] = 2 * math.pi * data.date.dt.dayofweek / 7
    data['dayofweek_cos'] = np.cos(data['dayofweek_norm'])
    data['dayofweek_sin'] = np.sin(data['dayofweek_norm'])
    data = data.drop('dayofweek_norm', axis=1)
    
    data['dayofweek'] = data.date.dt.dayofweek
    data = pd.get_dummies(data, columns=['dayofweek'])
    return data


def make_daily_data(data, idx):
    op = {f'open_{d}': 'first' for d in idx}
    cl = {f'close_{d}': 'last' for d in idx}
    hi = {f'high_{d}': 'max' for d in idx}
    lo = {f'low_{d}': 'min' for d in idx}
    vo = {f'volume_{d}': 'mean' for d in idx}
    re = {f'return_{d}': lambda x: np.prod(x+1)-1 for d in idx}
    # lr = {f'log_return_{d}': 'sum' for d in idx}

    groupdict = op
    for i in (cl, hi, lo, vo, re):
        groupdict.update(i)
    
    data_day = data.groupby(data.date.dt.date).aggregate(groupdict)
    for i in idx:
        data_day[f'volatility_{i}'] = data.groupby(data.date.dt.date).aggregate({f'return_{i}': lambda x: np.sqrt((x**2).sum())})
        data_day[f'spread_{i}'] = (data_day[f'high_{i}'] - data_day[f'low_{i}']) / data_day[f'low_{i}']
    data_day.index = pd.DatetimeIndex(data_day.index)
    
    data_day['dayofweek'] = data_day.index.dayofweek
    data_day = pd.get_dummies(data_day, columns=['dayofweek'])
    
    return data_day


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
            means.append(mean(df.iloc[:train_size][column]))
            stds.append(std(df.iloc[:train_size][column]))
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
        y = self.data[idx:idx+self.seq_length, [self.y_idx]].float()
        target = self.data[idx+self.seq_length:idx+self.seq_length+self.steps, self.y_idx].float()#.squeeze(0)
        if self.task == 'classification':
            target = (target > y[-1]).float()
        return (
            x, # x lags
            y, # y lags
            (target if isinstance(self.y_idx, int) else target.squeeze(0)) # target
        )

    
def take_vars(data, idx, var, add=''):
    take = [f'{v}_{i}' for i in idx for v in var]
    take = list(set(take)&set(data.columns))
    take = take + [c for c in data.columns if c.split('_')[0] in add]
    return data[take].copy()

    
class DataCollection():
    def __init__(self, data, config):
        idx = config['idx'].split(',')
        var = config['var'].split(',')
        add = config['add'].split(',')
        data = take_vars(data, idx, var, add)
        self.config = config
        self.data = data
        self.norm_stats = calculate_norm_stats(self.data, train_size=0.8, exclude=['time', 'dayofweek'], only_std=['return'], robust=True)
        
    @property
    def datasets(self):
        y = self.config['target'].split(',')
        if len(y)==1:
            y=y[0]
        full_dataset = TickerDataset(
            self.data, 
            y=y, 
            seq_length=self.config['seq_length'], 
            norm_stats=self.norm_stats, 
            steps=self.config['steps'], 
            task=self.config['task'])
        train_dataset, val_dataset, test_dataset = sequential_split(full_dataset, splits=[0.8, 0.1, 0.1])
        self.config['input_size'] = self.data.shape[1]
        return full_dataset, train_dataset, val_dataset, test_dataset
    @property
    def dataloaders(self):
        full_dataset, train_dataset, val_dataset, test_dataset = self.datasets
        full_loader = DataLoader(full_dataset, batch_size=self.config['batch_size'], shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)
        return full_loader, train_loader, val_loader, test_loader

    
def plot_preds(y_pred, y_true, splits=None):
    fig = plt.figure(figsize=(24,6))
    plt.plot(y_true)
    plt.plot(y_pred)
    if splits is not None:
        plt.axvline(x=splits[0], color='r', linestyle='--')
        plt.axvline(x=splits[1], color='r', linestyle='--')
    return fig


def plot_loss(train_loss, valid_loss, same_axis=True):
    if same_axis == True:
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('train loss')
        ax1.plot(train_loss)

        ax2 = ax1.twinx()
        ax2.set_ylabel('valid loss')
        ax2.plot(valid_loss, color='orange')

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