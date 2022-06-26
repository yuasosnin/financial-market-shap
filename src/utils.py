from typing import *
from pandas._typing import FilePath

import os
import copy
import functools
import string

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torch._utils import _accumulate
import pytorch_lightning as pl
from captum.attr import GradientShap


groupdict = dict(
    indexes = ['IMOEX', 'RTSI'],
    commodities = ['GC', 'NG', 'BZ'],
    shares = ['GAZP', 'SBER', 'LKOH', 'GMKN', 'NVTK', 'MGNT', 'ROSN', 'TATN', 'MTSS'],
    sectors = ['MOEXOG', 'MOEXEU', 'MOEXTL', 'MOEXMM', 'MOEXFN', 'MOEXCN', 'MOEXCH'],
    foreign = ['UKX', 'INX', 'NDX'],
    bonds = ['1W', '1M', '6M', '1Y', '3Y', '5Y', '10Y', '20Y'],
    currencies = ['USD', 'EUR']
)


def multi_merge(
    *args, 
    on: Optional[Union[List[str], str]] = None, 
    how: Literal['left', 'right', 'outer', 'inner', 'cross'] = 'outer'
) -> pd.DataFrame:
    '''Merge multiple data frames of common structure with pd.merge'''
    return functools.reduce(lambda df1, df2: pd.merge(df1, df2, on=on, how=how), args)


def sequential_split(dataset: Dataset, lengths: Sequence[int]) -> List[Subset]:
    '''
    A function similar to torch.utils.data.random_split
    Split a dataset sequentially into new datasets of given lengths.

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    '''
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):
        raise ValueError('Sum of input lengths does not equal the length of the input dataset!')

    return [Subset(dataset, range(offset - length, offset)) 
            for offset, length in zip(_accumulate(lengths), lengths)]


def take_vars(
    data: pd.DataFrame, 
    idx: Sequence[str], 
    var: Sequence[str], 
    add: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    '''
    Take specified columns from a data frame.
    
    Args:
        var: varable names before underscore, like return if return_imoex
        idx: index names after underscore, like imoex in return_imoex
        add: additional column names to be taken
    '''
    
    take = [f'{v}_{i}' for i in idx for v in var]
    take = [c for c in data.columns if c in take]
    if add is not None:
        take = take + [c for c in data.columns if c.split('_')[0] in add]
    return data[take].copy()


def process_predcit(pred: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Concatenate outputs of Trainer().predict(...)'''
    y_true, y_pred = [], []
    for t, p in pred:
        y_true.append(t[:,0].flatten())
        y_pred.append(p[:,0].flatten())
    return torch.cat(y_true), torch.cat(y_pred)


def l_out(l_in: int, kernel_size: int, padding: int = 0, dilation: int = 1, stride: Optional[int] = None) -> int:
    '''
    Calculate sequence length for outputs of nn.Conv1d or nn.MaxPool1d layers.
    
    Args:
        l_in: sequence length of input
        kernel_size, padding, dilation, stride: layer parameters
    '''
    if stride is None:
        stride = kernel_size
    return int(((l_in + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1)


def count_params(model: nn.Module) -> int:
    '''Return the number of trainable weights of a torch model.'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def read_logs(path: FilePath) -> dict[str: pd.DataFrame]:
    '''Read metrics.csv produced by pytorch_lightning.loggers.CSVLogger into a dictionary.'''
    logs = pd.read_csv(path)
    return {c: logs[c].dropna().reset_index(drop=True) for c in logs.columns}


def read_model_logs(model_name: str, split: Literal['val', 'test'] = 'val') -> pd.DataFrame:
    '''
    Read a folder of results.csv with the following structure:
    -logs
        -model_name
            -period_0
                -version_0
                    -results.csv
                -version_1
                    -results.csv
                ...
    Return a pandas DataFrame with MSE for periods and versions, calculated on specified split.
    
    Args:
        model_name: name of folder to read results from
        split: whether to return validation or test MSE
    '''
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


def res_table(logs_table: pd.DataFrame, model_name: str = 'model') -> pd.DataFrame:
    '''Return table with test MSE of models across variants, based on validation.'''
    zero_mse = logs_table['test_zero_mse'].unique()
    idx_min = logs_table.groupby('period')['val_mse'].idxmin()
    min_table = logs_table.iloc[idx_min]
    return min_table.reset_index()[
        [f'test_zero_mse', f'test_mse']
    ].T.rename({f'test_zero_mse': 'naive_zero', f'test_mse': model_name})


class PrintMetricsCallback(pl.callbacks.Callback):
    '''
    A Callback to print metrics into console.
    '''
    
    def __init__(self, metrics: Optional[Sequence[str]] = None) -> None:
        self.epoch = 0
        self.metrics = metrics

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics_dict = copy.copy(trainer.callback_metrics)
        metrics = self.metrics if self.metrics is not None else metrics_dict.keys()
        print('epoch:', self.epoch)
        for metric in metrics:
            if metric in metrics_dict:
                print(f'{metric}:', metrics_dict[metric].item())
        print('-'*80)
        self.epoch += 1
