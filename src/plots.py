from typing import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .waterfall_chart import plot as waterfall


def plot_predictions(
    y_pred, y_true, 
    splits=None, 
    ax=None, 
    legend=True, 
    plot_kwargs={}, 
    axvline_kwargs={}
):
    if ax is None:
        ax = plt.gca()
    
    ax.plot(y_true, **plot_kwargs)
    ax.plot(y_pred, **plot_kwargs)
    if legend:
        ax.legend(('True', 'Pred'))
    if splits is not None:
        for split in splits:
            ax.axvline(x=split, **(dict(color='r', linestyle='--') | axvline_kwargs))
    return ax


def plot_cumsum(
    y_pred, y_true, 
    ax=None, 
    legend=True, 
    colors=('C0', 'C1'), 
    plot_kwargs={}
):
    if ax is None:
        ax = plt.gca()
        
    cumsum = np.cumsum((y_true - y_pred)**2)
    cumsum_zero = np.cumsum((y_true)**2)
    ax.plot(cumsum_zero, **(dict(color=colors[0], label='Zero') | plot_kwargs))
    ax.plot(cumsum, **(dict(color=colors[1], label='Model') | plot_kwargs))
    if legend:
        ax.legend()
    return ax


def plot_loss(
    train_loss, valid_loss, 
    ax=None,
    same_axis=True,
    legend=True,
    colors=('C0', 'C1'), 
    plot_kwargs={}
):
    if ax is None:
        ax = plt.gca()
    
    if same_axis:
        ax2 = ax.twinx()

        ax.plot(train_loss, **(dict(color=colors[0], label='Train') | plot_kwargs))
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Train loss')

        ax2.plot(valid_loss, **(dict(color=colors[1], label='Valid') | plot_kwargs))
        ax2.set_ylabel('Valid loss')

        line, label = ax.get_legend_handles_labels()
        line2, label2 = ax2.get_legend_handles_labels()
        if legend:
            ax.legend(line+line2, label+label2, loc=2)
    else:
        ax.plot(train_loss, **(dict(color=colors[0], label='Train') | plot_kwargs))
        ax.plot(valid_loss, **(dict(color=colors[1], label='Valid') | plot_kwargs))
        if legend:
            ax.legend(loc=2)
    return ax


def plot_result_box(
    table, 
    ax=None, 
    split='val', 
    box_kwargs={}, 
    plot_kwargs={}
):
    zero_mse = table[f'{split}_zero_mse'].unique()
    a = table.pivot(index='period', columns='version')[f'{split}_mse'].values
    
    if ax is None:
        ax = plt.gca()
    
    ax.boxplot(a.T, **box_kwargs);
    ax.plot(range(1,a.shape[0]+1), zero_mse, **(dict(ls='o-', linewidth=2) | plot_kwargs))
    return ax


def plot_result_min(
    table, 
    ax=None, 
    split='val', 
    plot_kwargs={}
):
    zero_mse = table[f'{split}_zero_mse'].unique()
    idx_min = table.groupby('period')[f'val_mse'].idxmin()
    min_table = table.iloc[idx_min]
    
    if ax is None:
        ax = plt.gca()
    
    ax.plot(
        range(1, min_table.shape[0]+1), 
        min_table[f'{split}_zero_mse'], 
        **(dict(ls='o-', linewidth=2) | plot_kwargs)
    )
    ax.plot(
        range(1, min_table.shape[0]+1), 
        min_table[f'{split}_mse'], 
        **(dict(ls='o-', linewidth=2) | plot_kwargs)
    )
    return ax
