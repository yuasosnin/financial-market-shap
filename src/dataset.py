from typing import *

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from .utils import *


def calculate_norm_stats(
    df: pd.DataFrame,
    train_size: Optional[int] = None,
    exclude: Optional[Sequence[str]] = None,
    only_std: Optional[Sequence[str]] = None,
) -> List[List[float]]:
    '''
    Caclulate means and stds for a datset, on specified training subset.

    Args:
        df: data to calculate stats
        train_size: index in df, calculates means and std only for df[:train_size],
            defaults to all dataset
        exclude: columns to exclude from normalization, returns mean 0 and std 1 for them
        only_std: columns to calculate only std, and return mean=0
    '''

    exclude = exclude or []
    only_std = only_std or []
    train_size = train_size or df.shape[0]

    means = []
    stds = []
    for column in df.columns:
        if column.split('_')[0] in exclude or column in exclude:
            means.append(0)
            stds.append(1)
        elif column.split('_')[0] in only_std or column in only_std:
            means.append(0)
            stds.append(np.std(df.iloc[:train_size][column]))
        else:
            means.append(np.mean(df.iloc[:train_size][column]))
            stds.append(np.std(df.iloc[:train_size][column]))
    return [means, stds]


class TickerDataset(Dataset):
    '''
    A torch Dataset for data used in the project.

    Args:
        df_x: pandas DataFrame of all data
        y: either column name in df_x, sequence of column names,
            index or sequence of indexes to be used as targets
        seq_length: window size, 1st dimension in output tensors
        norm_stats: normalization statistics; list of 2 lists,
            means and stds for each column in df_x
    '''

    def __init__(
        self,
        df_x: pd.DataFrame,
        y: Union[str, Sequence[str], int, Sequence[int]],
        seq_length: int = 10,
        norm_stats: Optional[Sequence] = None
    ) -> None:

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

        self.x_idx = [
            i for i in range(
                self.data.shape[1]) if (
                i not in (
                    self.y_idx if isinstance(
                        self.y_idx,
                        list) else [
                        self.y_idx]))]

    def __len__(self) -> int:
        return self.data.shape[0] - self.seq_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        x = self.data[idx:idx + self.seq_length, self.x_idx].float()
        if isinstance(self.y_idx, int):
            y = self.data[idx:idx + self.seq_length, [self.y_idx]].float()
        else:
            y = self.data[idx:idx + self.seq_length, self.y_idx].float()
        target = self.data[idx + self.seq_length, [self.y_idx]].float()
        return x, y, target


class MyDataModule(pl.LightningDataModule):
    '''
    Pytorch Lightning wrapper for TickerDataset.
    '''

    def __init__(
        self,
        data: pd.DataFrame,
        target: Union[str, Sequence[str]],
        seq_length: int = 5,
        batch_size: int = 32,
        num_workers: int = 1,
        split_lengths: Optional[Sequence[int]] = None,
    ) -> None:

        super().__init__()
        self.data = data
        self.target = target
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        if split_lengths is None:
            self.split_lengths = [len(data) - seq_length - 2 * 5, 5, 5]
        else:
            self.split_lengths = split_lengths
        self.norm_stats = calculate_norm_stats(
            self.data, train_size=self.split_lengths[0])
        self.full_dataset = TickerDataset(
            self.data,
            y=self.target,
            seq_length=self.seq_length,
            norm_stats=self.norm_stats)
        self.train_dataset, self.val_dataset, self.test_dataset = sequential_split(
            self.full_dataset, lengths=self.split_lengths)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, shuffle=True,
                          batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, shuffle=False,
                          batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, shuffle=False,
                          batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.full_dataset, shuffle=False,
                          batch_size=self.batch_size, num_workers=self.num_workers)
