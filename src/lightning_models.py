from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pytorch_forecasting

from .utils import *
from .models import *


class DefaultLightningModule(pl.LightningModule):
    def forward(self, x, y):
        x = torch.cat((x, y), dim=2)
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y, target = batch
        output = self.forward(x, y)
        loss = self.criterion(output, target)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, target = batch
        output = self.forward(x, y)
        loss = self.criterion(output, target)
        self.log('valid_loss', loss.item(), on_step=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y, target = batch
        output = self.forward(x, y)
        return target, output

    def configure_optimizers(self):
        if self.hparams.optimizer == 'Ranger':
            optimizer = getattr(
                pytorch_forecasting.optim, self.hparams.optimizer
            )(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        else:
            optimizer = getattr(
                torch.optim, self.hparams.optimizer
            )(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)

        if self.hparams.scheduler in {'CosineAnnealingLR', 'CosineAnnealingWarmRestarts'}:
            scheduler = getattr(
                torch.optim.lr_scheduler, self.hparams.scheduler
            )(optimizer, self.hparams.T_max)
        else:
            scheduler = getattr(
                torch.optim.lr_scheduler, self.hparams.scheduler
            )(optimizer, self.hparams.gamma)
        return [optimizer], [scheduler]


class LitVanillaLSTM(DefaultLightningModule):
    def __init__(self, model_params, **hparams):
        super().__init__()
        self.model = VanillaLSTM(**model_params)
        self.hparams['model'] = self.model.__class__.__name__
        self.save_hyperparameters()
        self.criterion = getattr(torch.nn, self.hparams.loss)()


class LitCNN(DefaultLightningModule):
    def __init__(self, model_params, **hparams):
        super().__init__()
        self.model = CNN(**model_params)
        self.hparams['model'] = self.model.__class__.__name__
        self.save_hyperparameters()
        self.criterion = getattr(torch.nn, self.hparams.loss)()


class LitCNNLSTM(DefaultLightningModule):
    def __init__(self, model_params, **hparams):
        super().__init__()
        self.model = CNNLSTM(**model_params)
        self.hparams['model'] = self.model.__class__.__name__
        self.save_hyperparameters()
        self.criterion = getattr(torch.nn, self.hparams.loss)()
