import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm.notebook import tqdm
from pprint import pprint
import json, datetime
from .utils import *


class Trainer(nn.Module):
    def __init__(self, model, criterion=None, metric=None, device=None, task='regression'):
        super().__init__()
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.task = task
        self.logs = {}
        self.epochs = 0
        self.optimizer = None
        self.scheduler = None
        
        self.model = model.to(self.device)
        self.criterion = criterion
        self.metric = metric
    
    def forward(self, x, y):
        output =  self.model.forward(x, y)
        if self.task == 'classification':
            return torch.sigmoid(output)
        return output

    def log(self, name, value):
        if name not in self.logs:
            self.logs[name] = np.array([])
        if isinstance(value, float) or isinstance(value, int):
            self.logs[name] = np.append(self.logs[name], value)
        elif isinstance(value, np.ndarray):
            self.logs[name] = np.concatenate((self.logs[name], value))
            
    def save_logs(self, name):
        with open(f'logs/{name}.json', 'w') as fp:
            todump = {}
            todump['training_params'] = {
                'epochs': self.epochs,
                'optimizer': self.optimizer.__class__.__name__,
                'scheduler' : (self.scheduler.__class__.__name__ if self.scheduler is not None else None)
            }
            todump['model_params'] = self.model.params
            todump['optimizer_params'] = self.optimizer.state_dict()['param_groups']
            if self.scheduler is not None:
                todump['scheduler_params'] = self.scheduler.state_dict()
            todump['logs'] = self.logs
            json.dump(todump, fp, indent=4, cls=NumpyEncoder)

        torch.save(self.model.state_dict(), f'logs/{name}.pth')
    
    @staticmethod
    def _epoch_mean(array, loader_len):
        return array[-loader_len:].sum() / loader_len

    def _prepare_batch(self, batch):
        x, y, target  = batch
        x = x.to(self.device)
        y = y.to(self.device)
        target = target.to(self.device)
        return x, y, target
    
    def training_step(self, batch):
        x, y, target  = self._prepare_batch(batch)
        output = self.forward(x, y)
        loss = self.criterion(output, target)

        self.log('train_step_loss', loss.item())
        self.log('train_step_metric', float(self.metric(target.flatten().cpu(), output.detach().flatten().cpu())))
        return loss
    
    def validation_step(self, batch):
        x, y, target  = self._prepare_batch(batch)
        output = self.forward(x, y)
        loss = self.criterion(output, target)
        
        self.log('valid_step_loss', loss.item())
        self.log('valid_step_metric', float(self.metric(target.flatten().cpu(), output.detach().flatten().cpu())))
        return loss
    
    def prediction_step(self, batch):
        x, y, _ = self._prepare_batch(batch)
        output = self.forward(x, y)
        return output.cpu()
    
    def fit(self, train_loader, val_loader, epochs=10, verbose=True, save_logs=False, stopper=None):
        if verbose: tqdm_ = tqdm
        else: tqdm_ = lambda x: x
        if hasattr(self.model, 'params'):
            pprint(self.model.params)
        self.starttime = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        
        for epoch in tqdm_(range(epochs)):
            if verbose: print(f'{epoch+1} epoch:')
            self.epochs += 1

            self.train()
            for batch in tqdm_(train_loader):
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.log(
                'train_epoch_loss', 
                self._epoch_mean(self.logs['train_step_loss'], len(train_loader)))
            self.log(
                'train_epoch_metric', 
                self._epoch_mean(self.logs['train_step_metric'], len(train_loader)))
            if verbose: 
                print(f"train loss {self.logs['train_epoch_loss'][-1]}")
                print(f"train {self.metric.__name__} {self.logs['train_epoch_metric'][-1]}")
                
            if val_loader is not None:
                self.eval()
                for batch in tqdm_(val_loader):
                    loss = self.validation_step(batch)
                self.log(
                    'valid_epoch_loss', 
                    self._epoch_mean(self.logs['valid_step_loss'], len(val_loader)))
                self.log(
                    'valid_epoch_metric', 
                    self._epoch_mean(self.logs['valid_step_metric'], len(val_loader)))
                if verbose: 
                    print(f"valid loss {self.logs['valid_epoch_loss'][-1]}")
                    print(f"valid {self.metric.__name__} {self.logs['valid_epoch_metric'][-1]}")
            if verbose: print('-'*80)
            
        if save_logs:
            self.save_logs(self.starttime)
    
    def predict(self, data_loader, verbose=True):
        if verbose: tqdm_ = tqdm
        else: tqdm_ = lambda x: x
        ret = {
            'y_true': np.array([]),
            'y_pred': np.array([])
        }
        self.eval()
        for batch in tqdm_(data_loader):
            X, y, target  = batch
            output = self.prediction_step(batch)
            ret['y_pred'] = np.concatenate((ret['y_pred'], output.detach().numpy().flatten()))
            ret['y_true'] = np.concatenate((ret['y_true'], target.cpu().numpy().flatten()))
        return ret

    def plot_loss(self, same_axis=True):
        if same_axis == True:
            fig, ax1 = plt.subplots()
            ax1.set_xlabel('epochs')
            ax1.set_ylabel('train loss')
            ax1.plot(self.logs['train_epoch_loss'])

            ax2 = ax1.twinx()
            ax2.set_ylabel('valid loss')
            ax2.plot(self.logs['valid_epoch_loss'], color='orange')

            fig.tight_layout()
            return fig
        else:
            plt.plot(self.logs['train_epoch_loss'])
            plt.plot(self.logs['valid_epoch_loss'], color='orange')
            return None
    
    def plot(self, x):
        return plt.plot(self.logs[x])
    