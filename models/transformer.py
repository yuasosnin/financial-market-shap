import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm.notebook import tqdm
import math
from .utils import *
# from .trainer import Trainer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class VanillaTransformer(nn.Module):
    def __init__(self, input_size, seq_length, num_layers=1, nhead=1, dropout=0):
        super().__init__()
        # self.model_type = 'Transformer'
        
        self.input_size = input_size
        self.seq_length = seq_length
        self.nhead = self.input_size
        self.dropout = dropout
        self.num_layers = num_layers
        
        self.params = {
            'model': 'VanillaTransformer',
            'input_size': input_size,
            'seq_length': seq_length,
            'num_layers': num_layers,
            'dropout': dropout
        }
        
        # self.src_mask = None
        # self.pos_encoder = PositionalEncoding(self.input_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.nhead, dropout=self.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)        
        self.decoder = nn.Linear(self.input_size, 1)
        self.src_mask = self._generate_square_subsequent_mask(self.seq_length)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # if self.src_mask is None or self.src_mask.size(0) != len(x):
        #     device = x.device
        #     mask = self._generate_square_subsequent_mask(len(x)).to(device)
        #     self.src_mask = mask
        # src = self.pos_encoder(src)
        output = self.transformer_encoder(x, self.src_mask)
        output = self.decoder(output)
        return output.squeeze(2)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    
# class TransformerTrainer(Trainer):
#     def forward(self, x, y):
#         x = torch.cat((x, y), dim=2)
#         output =  self.model.forward(x)
#         if self.task == 'classification':
#             return torch.sigmoid(output)
#         return output
    
#     def _prepare_batch(self, batch):
#         x, y, target  = batch
#         x = x.to(self.device)
#         y = y.to(self.device)
#         target = target.to(self.device)
#         target = torch.cat([y[:,1:,:].flatten(1), target], dim=1)
#         return x, y, target
    