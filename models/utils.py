import os

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import mean_squared_error as mse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader


def columns_suffix(df, name):
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df.set_index('date', drop=True)
    df.columns = [f'{c}_{name}' for c in df.columns]
    df = df.reset_index()
    return df

def multi_merge(*args, on=None, how='outer'):
    from functools import reduce
    return reduce(lambda df1, df2: pd.merge(df1, df2, on=on, how=how), args)

def robust_mean(x, p=0.05):
    low = x.quantile(p)
    high = x.quantile(1-p)
    return x[(x>low) & (x<high)].mean()

def robust_std(x, p=0.05):
    low = x.quantile(p)
    high = x.quantile(1-p)
    return x[(x>low) & (x<high)].std()

def logreturn(s):
    return np.log(s) - np.log(s.shift(1))


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


def convert_to_lagged_df(dataset, colnames=None):
    x,y,target = dataset[0]
    num_lags = y.shape[0]
    num_predictors = x.shape[1]
    if colnames is None:
        colnames = [f'x_{i}' for i in range(num_predictors)]
    assert len(colnames) == num_predictors
    gen = (np.concatenate([target.numpy(), y.flatten().flip(0).numpy(), x.flip(0).flatten().numpy()]) for (x,y,target) in dataset)
    return pd.DataFrame(gen, columns=['y_target']+[f'y_lag{i+1}' for i in range(num_lags)]+[f'{colnames[j]}_lag{i+1}' for i in range(num_lags) for j in range(num_predictors)])


def l_out(l_in, kernel_size, padding=0, dilation=1, stride=None):
    if stride is None:
        stride = kernel_size
    return int(((l_in + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1)


def number_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_params(model):
    return {k:v for k,v in model.__dict__.items() if k.split('_')[0] not in ('', 'training')}


def read_all_logs(model_name, split='val'):
    errors_table = {'period':[], 'version':[], 'val_mse':[], 'val_zero_mse':[], 'test_mse':[], 'test_zero_mse':[]}
    n_splits = max([int(f.split('_')[-1]) for f in os.listdir(f'logs/model_{model_name}')])+1
    n_versions = max([int(f.split('_')[-1]) for f in os.listdir(f'logs/model_{model_name}/period_0')])+1

    for p in tqdm(range(n_splits)):
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
    return min_table.reset_index()[[f'test_zero_{metric}', f'test_{metric}']].T.rename({f'test_zero_{metric}': 'naive_zero', f'test_{metric}': model_name})


class PrintMetricsCallback(Callback):
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
        
           
class PeriodicCheckpoint(Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """
    '''https://github.com/PyTorchLightning/pytorch-lightning/issues/2534'''

    def __init__(
        self,
        every_n_train_steps=None,
        every_n_epochs=None,
        filename=None,
        dirpath=None
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        assert (every_n_train_steps is not None) or (every_n_epochs is not None)
        self.every_n_train_steps = every_n_train_steps
        self.every_n_epochs = every_n_epochs
        self.filename = filename
        self.dirpath = dirpath
        
    def on_batch_end(self, trainer, *args):
        """ Check if we should save a checkpoint after every train batch """
        from string import Template
        epoch = trainer.current_epoch
        step = trainer.global_step
        if (step % (self.every_n_train_steps or 0.1) == 0) or (epoch % (self.every_n_epochs or 0.1) == 0):
            if self.filename is None:
                filename = f'{epoch}-{step}.ckpt'
            else:
                filename = Template(self.filename.replace('{', '$').replace('}', '')).substitute(epoch=epoch, step=step)
            if self.dirpath is None:
                ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            else:
                ckpt_path = os.path.join(self.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


def process_predcit(pred):
    y_true, y_pred = [], []
    for t, p in pred:
        y_true.append(t[:,0].flatten())
        y_pred.append(p[:,0].flatten())
    return torch.cat(y_true), torch.cat(y_pred)


def read_logs(path):
    logs = pd.read_csv(path)
    return {c: logs[c].dropna().reset_index(drop=True) for c in logs.columns}
