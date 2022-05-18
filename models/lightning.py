import copy, os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_forecasting
from .utils import *
from .lstm import *
from .darnn import *


class MyDataModule(pl.LightningDataModule):
    def __init__(self, data, target, seq_length=5, batch_size=32, num_workers=1, splits=[0.8, 0.1, 0.1], norm_stats=None):
        super().__init__()
        self.data = data
        self.target = target
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splits = splits
        self.norm_stats = norm_stats

    def setup(self, stage=None):
        self.norm_stats = calculate_norm_stats(self.data, train_size=self.splits[0], exclude=['time','dayofweek'], robust=True)
        # self.denoise = lambda x: denoise_wavelet(x, method='BayesShrink', mode='soft', wavelet_levels=3, wavelet='Haar')
        self.full_dataset = TickerDataset(self.data, y=self.target, seq_length=self.seq_length, norm_stats=self.norm_stats)
        self.train_dataset, self.val_dataset, self.test_dataset = sequential_split(self.full_dataset, splits=self.splits)
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.full_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)

class MyLightningModule(pl.LightningModule):
    def __init__(self, model, **hparams):
        super().__init__()
        self.hparams['model'] = model.__class__.__name__
        self.hparams['model_params'] = get_model_params(model)
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.criterion = getattr(torch.nn, self.hparams.loss)()
    
    def forward(self, x, y):
        if self.model.__class__.__name__ == 'DARNN':
            output, x_attention, t_attention = self.model.forward(x, y)
        else:
            x = torch.cat((x, y), dim=2)
            output =  self.model.forward(x)
        return output
    
    def training_step(self, batch, batch_idx):
        x, y, target = batch
        if self.model.__class__.__name__ == 'VanillaTransformer':
            target = torch.cat([y[:,1:,:].flatten(1), target], dim=1)
        output = self.forward(x, y)
        loss = self.criterion(output, target)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, target  = batch
        if self.model.__class__.__name__ == 'VanillaTransformer':
            target = torch.cat([y[:,1:,:].flatten(1), target], dim=1)
        output = self.forward(x, y)
        loss = self.criterion(output, target)
        self.log('valid_loss', loss.item(), on_step=True, on_epoch=True)
        return loss
        
    def predict_step(self, batch, batch_idx):
        x, y, target  = batch
        output = self.forward(x, y)
        if self.model.__class__.__name__ == 'VanillaTransformer':
            output = output[:,-1][:,None]
        return target, output
    
    def configure_optimizers(self):
        if self.hparams.optimizer == 'Ranger':
            optimizer = getattr(pytorch_forecasting.optim, self.hparams.optimizer)(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        else:
            optimizer = getattr(torch.optim, self.hparams.optimizer)(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        if self.hparams.scheduler in ('CosineAnnealingLR', 'CosineAnnealingWarmRestarts'):
            scheduler = getattr(torch.optim.lr_scheduler, self.hparams.scheduler)(optimizer, self.hparams.T_max)
        else:
            scheduler = getattr(torch.optim.lr_scheduler, self.hparams.scheduler)(optimizer, self.hparams.gamma)
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