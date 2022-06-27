from typing import *

import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from .utils import groupdict
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
            ax.axvline(x=split, 
                       **(dict(color='r', linestyle='--') | axvline_kwargs))
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

        ax.plot(train_loss, 
                **(dict(color=colors[0], label='Train') | plot_kwargs))
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Train loss')

        ax2.plot(valid_loss, 
                 **(dict(color=colors[1], label='Valid') | plot_kwargs))
        ax2.set_ylabel('Valid loss')

        line, label = ax.get_legend_handles_labels()
        line2, label2 = ax2.get_legend_handles_labels()
        if legend:
            ax.legend(line + line2, label + label2, loc=2)
    else:
        ax.plot(train_loss, 
                **(dict(color=colors[0], label='Train') | plot_kwargs))
        ax.plot(valid_loss, 
                **(dict(color=colors[1], label='Valid') | plot_kwargs))
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

    ax.boxplot(a.T, **box_kwargs)
    ax.plot(range(1, a.shape[0] + 1), zero_mse, 
            **(dict(marker='o', linewidth=2) | plot_kwargs))
    return ax


def plot_result_min(
    table,
    ax=None,
    split='val',
    colors=['C0', 'C1'],
    plot_zero=True,
    plot_kwargs={}
):
    zero_mse = table[f'{split}_zero_mse'].unique()
    idx_min = table.groupby('period')['val_mse'].idxmin()
    min_table = table.iloc[idx_min]

    if ax is None:
        ax = plt.gca()

    if plot_zero:
        ax.plot(
            range(1, min_table.shape[0] + 1),
            min_table[f'{split}_zero_mse'],
            **(dict(color=colors[0], marker='o', linewidth=2) | plot_kwargs)
        )
    ax.plot(
        range(1, min_table.shape[0] + 1),
        min_table[f'{split}_mse'],
        **(dict(color=colors[1], marker='o', linewidth=2) | plot_kwargs)
    )
    return ax


class AttributionPlotter():
    '''A class wrapping some plotting utilities for attributions.'''

    def __init__(
        self,
        feature_names: Sequence[str],
        attributions: torch.Tensor,
        inputs: torch.Tensor
    ) -> None:
        self.feature_names = feature_names
        self.attributions = attributions
        self.inputs = inputs

    @staticmethod
    def _groupper(x: str) -> str:
        '''Get which variable a column is of.'''
        var = x.split('_')[-1]
        return 'bonds' if var[0] in string.digits else var

    @property
    def _reverse_groupdict(self) -> dict:
        '''A reverse of global groupdict dictionary.'''
        reverse_groupdict = {}
        for k, v in groupdict.items():
            for x in v:
                reverse_groupdict[x.lower()] = k
        return reverse_groupdict

    def plot_local(
        self,
        day: int = 20,
        ax: Optional[plt.Axes] = None,
        fontsize: Optional[Union[int, float]] = None,
        line_kwargs: dict = {}
    ) -> plt.Axes:
        '''
        Create waterfall plot of local attributions for given day.

        Args:
            day: index to be plotted
            ax: matplotlib axes
            fontsize: size of bar annotations
            line_kwargs: a dictionary of additional parameters for plt.axhline
        '''

        df = pd.DataFrame({
            'feature_names': self.feature_names,
            'shap': self.attributions[day, :, :].sum(axis=0)
        })
        df = df.groupby(df['feature_names'].apply(self._groupper)).sum()

        if ax is None:
            ax = plt.gca()

        ax = waterfall(
            df.index,
            df.shap,
            ax=ax,
            sorted_value=True,
            formatting='{:,.2f}',
            green_color='C2',
            red_color='C3',
            blue_color='C1',
            net_label='prediction',
            rotation=90,
            fontsize=None,
            line_kwargs=(dict(linewidth=0.5, linestyle='-') | line_kwargs),
        )
        ax.set_ylabel('SHAP Value')
        return ax

    @property
    def _sorted_global_df(self):
        df = pd.DataFrame({
            'var': self.feature_names,
            'attr': torch.abs(self.attributions.sum(axis=1)).mean(axis=0)
        })
        return (
            df
            .groupby(df['var'].apply(self._groupper))
            .mean()
            .sort_values(by='attr', ascending=True)
        )

    def plot_global_bar(
        self,
        ax: Optional[plt.Axes] = None,
        bar_kwargs: dict = {}
    ) -> plt.Axes:
        '''
        Create barplot of absolute average global attribuitons for
        indexes, averaged across variables.

        Args:
            ax: matplotlib axes
            bar_kwargs: a dictionary of additional parameters for plt.bar
        '''

        df = self._sorted_global_df

        if ax is None:
            ax = plt.gca()

        ax.barh(df.index, df.attr, **(dict(color='grey') | bar_kwargs))
        ax.set_xlabel('Absolute Average Shap Value')
        return ax

    @property
    def _grouped_daily_df(self):
        df_shap = pd.DataFrame(self.attributions.mean(1)).T
        df_shap.index = self.feature_names
        df_shap = df_shap.groupby(self._groupper).sum().T.reset_index()

        df_input = pd.DataFrame(self.inputs.mean(1)).T
        df_input.index = self.feature_names
        df_input = df_input.groupby(self._groupper).sum().T.reset_index()

        return pd.merge(
            pd.melt(df_shap, 'index', var_name='feature', value_name='shap'),
            pd.melt(df_input, 'index', var_name='feature', value_name='input'),
        )

    def plot_global_scatter(
        self,
        ax: Optional[plt.Axes] = None,
        scatter_kwargs: dict = {},
        line_kwargs: dict = {}
    ) -> plt.Axes:
        '''
        Create scatterplot of average global attribuitons for
        indexes, averaged across variables.

        Args:
            ax: matplotlib axes
            scatter_kwargs: a dictionary of additional parameters for sns.scatterplot
            line_kwargs: a dictionary of additional parameters
                for plt.axvline (zero vertial line)
        '''

        df = self._grouped_daily_df
        df_sort = self._sorted_global_df
        key = dict(zip(df_sort.index, df_sort.reset_index().index))
        df['sort_value'] = df['feature'].map(key)
        df = df.sort_values(['sort_value'], ascending=False)

        if ax is None:
            ax = plt.gca()

        sns.scatterplot(
            ax=ax,
            data=df,
            y='feature',
            x='shap',
            hue='input',
            **(dict(
                hue_norm=(-10, 10),
                palette='viridis',
                marker='o',
                linewidth=0,
                legend=False
            ) | scatter_kwargs)
        )
        ax.axvline(0, **(dict(color='black', linewidth=0.5) | line_kwargs))
        ax.set_ylabel(None)
        ax.set_xlabel('Shap Value')
        return ax

    def plot_indexes(
        self,
        ax: Optional[plt.Axes] = None,
        bar_kwargs: dict = {},
    ) -> plt.Axes:
        '''
        Create barplot of absolute average global attribuitons, for indexes, averaged across everything else.

        Args:
            ax: matplotlib axes
            bar_kwargs: a dictionary of additional parameters for plt.bar
        '''

        df = pd.DataFrame({
            'var': self.feature_names,
            'attr': torch.abs(self.attributions.sum(axis=1)).mean(axis=0)
        })
        toplot = df.groupby(
            df['var'].apply(
                lambda x: self._reverse_groupdict[x.split('_')[-1]])
        ).mean()

        if ax is None:
            ax = plt.gca()

        ax.barh(toplot.index, toplot.attr, **(dict(color='grey') | bar_kwargs))
        return ax

    def plot_variables(
        self,
        ax: Optional[plt.Axes] = None,
        bar_kwargs: dict = {},
    ) -> plt.Axes:
        '''
        Create barplot of absolute average global attribuitons, for variables, averaged across everything else.

        Args:
            ax: matplotlib axes
            bar_kwargs: a dictionary of additional parameters for plt.bar
        '''

        df = pd.DataFrame({
            'var': self.feature_names,
            'attr': torch.abs(self.attributions.sum(axis=1)).mean(axis=0)}
        )
        toplot = df.groupby(
            df['var'].apply(lambda x: x.split('_')[0])
        ).mean()

        if ax is None:
            ax = plt.gca()

        ax.barh(toplot.index, toplot.attr, **(dict(color='grey') | bar_kwargs))
        return ax

    def plot_global_all_stem(
        self,
        ax: Optional[plt.Axes] = None,
        stem_kwargs: dict = {},
    ) -> plt.Axes:
        '''
        Create stem plot of absolute average absolute global attribuitons, for all features, averaged only across days.

        Args:
            ax: matplotlib axes
            stem_kwargs: a dictionary of additional parameters for plt.stem
        '''

        toplot = torch.abs(self.attributions.sum(axis=1)).mean(axis=0)

        if ax is None:
            ax = plt.gca()

        ax.stem(
            np.array(self.feature_names)[toplot.argsort()],
            np.sort(toplot),
            orientation='horizontal',
            **(dict(
                markerfmt='ko',
                linefmt='black',
                basefmt='black'
            ) | stem_kwargs)
        )
        ax.set_ylim([-1, len(toplot)])
        return ax

    def plot_global_all_bar(
        self,
        ax: Optional[plt.Axes] = None,
        bar_kwargs: dict = {},
    ) -> plt.Axes:
        '''
        Create barplot of absolute average absolute global attribuitons, for all features, averaged only across days.

        Args:
            ax: matplotlib axes
            stem_kwargs: a dictionary of additional parameters for plt.bar
        '''

        toplot = torch.abs(self.attributions.sum(axis=1)).mean(axis=0)

        if ax is None:
            ax = plt.gca()

        ax.barh(
            np.array(self.feature_names)[toplot.argsort()],
            np.sort(toplot),
            **(dict(
                color='grey',
            ) | bar_kwargs)
        )
        ax.set_ylim([-1, len(toplot)])
        return ax
