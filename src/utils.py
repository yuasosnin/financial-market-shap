import os
import copy
import functools

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
import pytorch_lightning as pl
from captum.attr import GradientShap


groupdict = dict(
    indexes = ['IMOEX', 'RTSI'],
    commodities = ['GC', 'NG', 'BZ'],
    shares = ['GAZP', 'SBER', 'LKOH', 'GMKN', 'NVTK', 'MGNT', 'ROSN', 'TATN', 'MTSS', 'SNGS'],
    sectors = ['MOEXOG', 'MOEXEU', 'MOEXTL', 'MOEXMM', 'MOEXFN', 'MOEXCN', 'MOEXCH'],
    foreign = ['UKX', 'INX', 'NDX'],
    futures = ['MIX'],
    bonds = ['1W', '1M', '6M', '1Y', '3Y', '5Y', '10Y', '20Y'],
    currencies = ['USD', 'EUR']
)


def multi_merge(*args, on=None, how='outer'):
    return functools.reduce(lambda df1, df2: pd.merge(df1, df2, on=on, how=how), args)


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


def take_vars(data, idx, var, add=''):
    take = [f'{v}_{i}' for i in idx for v in var]
    take = [c for c in data.columns if c in take]
    take = take + [c for c in data.columns if c.split('_')[0] in add]
    return data[take].copy()


def process_predcit(pred):
    y_true, y_pred = [], []
    for t, p in pred:
        y_true.append(t[:,0].flatten())
        y_pred.append(p[:,0].flatten())
    return torch.cat(y_true), torch.cat(y_pred)


def l_out(l_in, kernel_size, padding=0, dilation=1, stride=None):
    if stride is None:
        stride = kernel_size
    return int(((l_in + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def read_logs(path):
    logs = pd.read_csv(path)
    return {c: logs[c].dropna().reset_index(drop=True) for c in logs.columns}


def read_model_logs(model_name, split='val'):
    errors_table = {'period':[], 'version':[], 'val_mse':[], 'val_zero_mse':[], 'test_mse':[], 'test_zero_mse':[]}
    n_splits = max([int(f.split('_')[-1]) for f in os.listdir(f'logs/model_{model_name}')])+1
    n_versions = max([int(f.split('_')[-1]) for f in os.listdir(f'logs/model_{model_name}/period_0')])+1

    for p in range(n_splits):
        for v in range(n_versions):
            y_true, y_pred = pd.read_csv(f'logs/model_{model_name}/period_{p}/version_{v}/results.csv').values.T
            errors_table['period'].append(p)
            errors_table['version'].append(v)
            errors_table['val_mse'].append(mse(y_true[-10:-5:], y_pred[-10:-5:]))
            errors_table['val_zero_mse'].append(mse(y_true[-10:-5:], np.zeros_like(y_pred[-10:-5:])))
            errors_table['test_mse'].append(mse(y_true[-5:], y_pred[-5:]))
            errors_table['test_zero_mse'].append(mse(y_true[-5:], np.zeros_like(y_pred[-5:])))
    table = pd.DataFrame(errors_table)
    return table


def res_table(logs_table, model_name='model', metric='mse'):
    zero_mse = logs_table['test_zero_mse'].unique()
    idx_min = logs_table.groupby('period')['val_mse'].idxmin()
    min_table = logs_table.iloc[idx_min]
    return min_table.reset_index()[
        [f'test_zero_{metric}', f'test_{metric}']
    ].T.rename({f'test_zero_{metric}': 'naive_zero', f'test_{metric}': model_name})


def attribute(x, model, baseline=None, method=GradientShap):
    x = torch.clone(x)
    x.requires_grad_()
    if baseline is None:
        baseline = torch.zeros_like(x)
    attributor = method(model)
    attr = attributor.attribute(x, baseline)
    return attr.detach()


class PrintMetricsCallback(pl.callbacks.Callback):
    def __init__(self, metrics=None):
        self.epoch = 0
        self.metrics = metrics

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics_dict = copy.copy(trainer.callback_metrics)
        metrics = self.metrics if self.metrics is not None else metrics_dict.keys()
        print('epoch:', self.epoch)
        for metric in metrics:
            if metric in metrics_dict:
                print(f'{metric}:', metrics_dict[metric].item())
        print('-'*80)
        self.epoch += 1


def _groupper(x):
    import string
    var = x.split('_')[-1]
    if var[0] in string.digits:
        return 'bonds'
    else:
        return var
