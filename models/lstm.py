import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm.notebook import tqdm
from .trainer import Trainer


class VanillaLSTM(nn.Module):
    def __init__(self, input_size, seq_length, output_size=1, steps=1, 
                 lstm_layers=1, lstm_hidden_size=64, bidirectional=False, 
                 fc_layers=None, fc_hidden_size=64, dropout=0, stateful=True):
        super().__init__()

        self.input_size = input_size
        self.seq_length = seq_length
        self.output_size = output_size
        self.steps = steps
        
        self.num_layers = lstm_layers
        self.hidden_size = lstm_hidden_size
        self.bidirectional = bidirectional
        
        self.fc_hidden_size = fc_hidden_size
        self.fc_layers = fc_layers
        self.dropout = dropout
        self.stateful = stateful
        
        self.params = {
            'model': 'VanillaLSTM',
            'input_size': input_size,
            'seq_length': seq_length,
            'output_size': output_size,
            'steps': steps,
            'num_layers': lstm_layers,
            'hidden_size': lstm_hidden_size,
            'bidirectional': bidirectional,
            'fc_hidden_size': fc_hidden_size,
            'fc_layers': fc_layers,
            'dropout': dropout,
            'stateful': stateful
        }
        
        self.lstm = nn.LSTM(
            bidirectional=self.bidirectional,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=(self.dropout if self.num_layers>1 else 0)
        )
        
        self.act = nn.Sigmoid()
        self.fc_bloc = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.Dropout(self.dropout),
            self.act
        )
        if self.fc_layers is not None:
            self.fc = nn.Sequential(
                nn.Linear(in_features=(self.bidirectional+1)*self.hidden_size*self.seq_length, out_features=self.hidden_size),
                nn.Dropout(self.dropout),
                self.act,
                *[self.fc_bloc]*self.fc_layers,
                nn.Linear(in_features=self.hidden_size, out_features=self.steps*self.output_size)
            )
        else:
            self.fc = nn.Linear(in_features=(self.bidirectional+1)*self.hidden_size*self.seq_length, out_features=self.steps*self.output_size)
            # self.fc = nn.Linear(in_features=(self.bidirectional+1)*self.hidden_size, out_features=self.steps*self.output_size)
            # self.fc = nn.Linear(in_features=(self.bidirectional+1)*self.hidden_size, out_features=self.steps*self.output_size)


    def forward(self, x):
        batch_size = x.shape[0]

        if self.stateful:
            h0 = torch.zeros((self.bidirectional+1)*self.num_layers, batch_size, self.hidden_size).requires_grad_()
            c0 = torch.zeros((self.bidirectional+1)*self.num_layers, batch_size, self.hidden_size).requires_grad_()
            x, (hn, cn) = self.lstm(x, (h0, c0))
        else:
            x, (hn, cn) = self.lstm(x)
        
        # print(x.shape)
        # print(hn.shape)
        # print(x[:, -1].shape)
        x = x.reshape(batch_size, -1)
        # x = x[:, -1]
        # x = self.act(x)
        x = self.fc(x)
        x = nn.Dropout(self.dropout)(x)
        return x
    
    
class EncoderDecoderLSTM(nn.Module):
    def __init__(self, input_size, seq_length, output_size=1, steps=1, 
                 encoder_layers=1, encoder_hidden_size=64, encoder_bidirectional=False,
                 decoder_layers=1, decoder_hidden_size=64, decoder_bidirectional=False,
                 fc_layers=0, fc_hidden_size=64, dropout=0):
        super().__init__()

        self.input_size = input_size
        self.seq_length = seq_length
        self.output_size = output_size
        self.steps = steps
        
        self.encoder_layers, self.decoder_layers = encoder_layers, decoder_layers
        self.encoder_hidden_size, self.decoder_hidden_size = encoder_hidden_size, decoder_hidden_size
        self.encoder_bidirectional, self.decoder_bidirectional = encoder_bidirectional, decoder_bidirectional
        
        self.fc_hidden_size = fc_hidden_size
        self.fc_layers = fc_layers
        self.dropout = dropout
        
        self.params = {
            'model': 'EncoderDecoderLSTM',
            'input_size': input_size,
            'seq_length': seq_length,
            'output_size': output_size,
            'steps': steps,
            'encoder_layers': encoder_layers,
            'encoder_hidden_size': encoder_hidden_size,
            'encoder_bidirectional': encoder_bidirectional,
            'decoder_layers': decoder_layers,
            'decoder_hidden_size': decoder_hidden_size,
            'decoder_bidirectional': decoder_bidirectional,
            'fc_hidden_size': fc_hidden_size,
            'fc_layers': fc_layers,
            'dropout': dropout
        }
        
        self.encoder = nn.LSTM(
            bidirectional=self.encoder_bidirectional,
            input_size=self.input_size,
            hidden_size=self.encoder_hidden_size,
            batch_first=True,
            num_layers=self.encoder_layers,
            dropout=(self.dropout if self.encoder_layers>1 else 0)
        )
        self.decoder = nn.LSTM(
            bidirectional=self.decoder_bidirectional,
            input_size=(self.encoder_bidirectional+1)*self.encoder_hidden_size,
            hidden_size=self.decoder_hidden_size,
            batch_first=True,
            num_layers=self.decoder_layers,
            dropout=(self.dropout if self.decoder_layers>1 else 0)
        )
        
        
        self.fc_bloc = nn.Sequential(
            nn.Linear(in_features=self.fc_hidden_size, out_features=self.fc_hidden_size),
            nn.Dropout(self.dropout),
            nn.ReLU()
        )
        if self.fc_layers is not None:
            self.fc = nn.Sequential(
                nn.Linear(in_features=(self.decoder_bidirectional+1)*self.decoder_hidden_size*self.seq_length, out_features=self.fc_hidden_size),
                nn.Dropout(self.dropout),
                nn.ReLU(),
                *[self.fc_bloc]*self.fc_layers,
                nn.Linear(in_features=self.fc_hidden_size, out_features=self.steps*self.output_size)
            )
        else:
            self.fc = nn.Linear(in_features=(self.decoder_bidirectional+1)*self.decoder_hidden_size*self.seq_length, out_features=self.steps*self.output_size)

    def forward(self, x):
        batch_size = x.shape[0]

        h0e = torch.zeros((self.encoder_bidirectional+1)*self.encoder_layers, batch_size, self.encoder_hidden_size).requires_grad_()
        c0e = torch.zeros((self.encoder_bidirectional+1)*self.encoder_layers, batch_size, self.encoder_hidden_size).requires_grad_()
        
        # h0d = torch.zeros((self.decoder_bidirectional+1)*self.decoder_layers, batch_size, self.decoder_hidden_size).requires_grad_()
        # c0d = torch.zeros((self.decoder_bidirectional+1)*self.decoder_layers, batch_size, self.decoder_hidden_size).requires_grad_()

        x, (hn, cn) = self.encoder(x, (h0e, c0e))
        x, (hn, cn) = self.decoder(x, (hn, cn))
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        return x


class LSTMTrainer(Trainer):
    def forward(self, x, y):
        x = torch.cat((x, y), dim=2)
        output =  self.model.forward(x)
        if self.task == 'classification':
            return torch.sigmoid(output)
        return output