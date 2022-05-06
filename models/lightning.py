import copy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *


class MyDataModule(pl.LightningDataModule):
    def __init__(self, data, target, seq_length=5, batch_size=32, num_workers=1):
        super().__init__()
        self.data = data
        self.target = target
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.norm_stats = calculate_norm_stats(self.data, train_size=0.8, exclude=['time','dayofweek'], only_std=['return_imoex'], robust=True)
        self.full_dataset = TickerDataset(self.data, y=self.target, seq_length=self.seq_length, norm_stats=self.norm_stats)
        self.train_dataset, self.val_dataset, self.test_dataset = sequential_split(self.full_dataset, splits=[0.8, 0.1, 0.1])
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.full_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)


# class MNISTKFoldDataModule(BaseKFoldDataModule):

#     train_dataset: Optional[Dataset] = None
#     test_dataset: Optional[Dataset] = None
#     train_fold: Optional[Dataset] = None
#     val_fold: Optional[Dataset] = None

#     def prepare_data(self) -> None:
#         # download the data.
#         MNIST(_DATASETS_PATH, transform=T.Compose([T.ToTensor(), T.Normalize(mean=(0.5,), std=(0.5,))]))

#     def setup(self, stage: Optional[str] = None) -> None:
#         # load the data
#         dataset = MNIST(_DATASETS_PATH, transform=T.Compose([T.ToTensor(), T.Normalize(mean=(0.5,), std=(0.5,))]))
#         self.train_dataset, self.test_dataset = random_split(dataset, [50000, 10000])

#     def setup_folds(self, num_folds: int) -> None:
#         self.num_folds = num_folds
#         self.splits = [split for split in KFold(num_folds).split(range(len(self.train_dataset)))]

#     def setup_fold_index(self, fold_index: int) -> None:
#         train_indices, val_indices = self.splits[fold_index]
#         self.train_fold = Subset(self.train_dataset, train_indices)
#         self.val_fold = Subset(self.train_dataset, val_indices)

#     def train_dataloader(self) -> DataLoader:
#         return DataLoader(self.train_fold)

#     def val_dataloader(self) -> DataLoader:
#         return DataLoader(self.val_fold)

#     def test_dataloader(self) -> DataLoader:
#         return DataLoader(self.test_dataset)

#     def __post_init__(cls):
#         super().__init__()


class MyLightningModule(pl.LightningModule):
    def __init__(self, model, **hparams):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.criterion = nn.MSELoss()
    
    def forward(self, x, y):
        x = torch.cat((x, y), dim=2)
        output =  self.model.forward(x)
        return output
    
    def training_step(self, batch, batch_idx):
        x, y, target = batch
        output = self.forward(x, y)
        loss = self.criterion(output, target)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, target  = batch
        output = self.forward(x, y)
        loss = self.criterion(output, target)
        self.log('valid_loss', loss.item(), on_step=True, on_epoch=True)
        return loss
        
    def predict_step(self, batch, batch_idx):
        x, y, target  = batch
        output = self.forward(x, y)
        return target, output
    
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, 'Adam')(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        scheduler = getattr(torch.optim.lr_scheduler, 'ExponentialLR')(optimizer, self.hparams.gamma)
        return [optimizer], [scheduler]
    
    
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


def process_predcit(pred):
    y_true, y_pred = [], []
    for t, p in pred:
        y_true.append(t.flatten())
        y_pred.append(p.flatten())
    return torch.cat(y_true), torch.cat(y_pred)


def read_logs(path):
    logs = pd.read_csv(path)
    return dict(
        train_loss_step = logs['train_loss_step'].dropna().reset_index(drop=True),
        train_loss_epoch = logs['train_loss_epoch'].dropna().reset_index(drop=True),
        valid_loss_step = logs['valid_loss_step'].dropna().reset_index(drop=True),
        valid_loss_epoch = logs['valid_loss_epoch'].dropna().reset_index(drop=True)
    )