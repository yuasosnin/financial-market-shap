import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm.notebook import tqdm
from .trainer import Trainer


class PreviousValueBaseline(nn.Module):
    def __init__(self, task='regression'):
        super().__init__()
        self.task = task
        
    def forward(self, x, y):
        if self.task == 'classification':
            return (y[:,-1,:] > y[:,-2,:]).float()
        else:
            return y[:,-1,:]

class RollingMeanBaseline(nn.Module):
    def __init__(self, task='regression'):
        super().__init__()
        
    def forward(self, x, y):
        if self.task == 'regression':
            return y.mean(dim=1)
        else:
            raise NotImplemented

class AlwaysOneBaseline(nn.Module):
    def __init__(self, task='regression'):
        super().__init__()
            
    def forward(self, x, y):
        return torch.ones(x.shape[0], 1, dtype=torch.float)