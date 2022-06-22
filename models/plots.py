import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Train loss')
        plot1 = ax1.plot(train_loss, label='Train')
        # ax1.legend()

        ax2 = ax1.twinx()
        ax2.set_ylabel('Valid loss')
        plot2 = ax2.plot(valid_loss, label='Valid', color='C1')
        # ax2.legend()

        # added these three lines
        plots = plot1+plot2
        labs = [l.get_label() for l in plots]
        ax1.legend(plots, labs, loc='upper right')
        # plt.title('Loss')

        fig.tight_layout()
        return fig
    else:
        plt.plot(train_loss)
        plt.plot(valid_loss, color='C1')
        return None

def plot_result_box(table, figsize=(15,5), split='val', **kwargs):
    zero_mse = table[f'{split}_zero_mse'].unique()
    a = table.pivot(index='period', columns='version')[f'{split}_mse'].values
    fig = plt.figure(figsize=figsize, **kwargs)
    plt.boxplot(a.T);
    plt.plot(range(1,a.shape[0]+1), zero_mse, 'o-', linewidth=2)
    plt.tight_layout()
    return fig

def plot_result_min(table, figsize=(15,5), split='val', **kwargs):
    zero_mse = table[f'{split}_zero_mse'].unique()
    idx_min = table.groupby('period')[f'val_mse'].idxmin()
    min_table = table.iloc[idx_min]
    fig = plt.figure(figsize=figsize, **kwargs)
    plt.plot(range(1,min_table.shape[0]+1), min_table[f'{split}_zero_mse'], 'o-', linewidth=2)
    plt.plot(range(1,min_table.shape[0]+1), min_table[f'{split}_mse'], 'o-', linewidth=2)
    plt.tight_layout()
    return fig
