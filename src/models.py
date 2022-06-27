from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *


class VanillaLSTM(nn.Module):
    def __init__(self, input_size, seq_length, output_size=1, steps=1,
                 num_layers=1, hidden_size=64, bidirectional=False,
                 fc_layers=None, fc_hidden_size=64, dropout=0, act=nn.Tanh):
        super().__init__()

        self.input_size = input_size
        self.seq_length = seq_length
        self.output_size = output_size
        self.steps = steps

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.fc_hidden_size = fc_hidden_size
        self.fc_layers = fc_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            bidirectional=self.bidirectional,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=(self.dropout if self.num_layers > 1 else 0)
        )

        self.act = act()
        self.drop = nn.Dropout(self.dropout)
        self.fc_bloc = nn.Sequential(
            nn.Linear(
                in_features=self.fc_hidden_size,
                out_features=self.fc_hidden_size),
            self.drop,
            self.act
        )
        if self.fc_layers is not None:
            self.fc = nn.Sequential(
                nn.Linear(
                    in_features=(
                        (self.bidirectional + 1)
                        * self.hidden_size
                        * self.seq_length),
                    out_features=self.fc_hidden_size),
                self.drop,
                self.act,
                *[self.fc_bloc] * self.fc_layers,
                nn.Linear(
                    in_features=self.fc_hidden_size,
                    out_features=self.steps * self.output_size)
            )
        else:
            self.fc = nn.Linear(
                in_features=(
                    (self.bidirectional + 1)
                    * self.hidden_size
                    * self.seq_length),
                out_features=self.steps * self.output_size)
        self.init_weights()

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight.data)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias.data, 0.0)
            elif isinstance(layer, nn.LSTM):
                for param in layer.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.constant_(param.data, 0)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
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

        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_bidirectional = encoder_bidirectional
        self.decoder_bidirectional = decoder_bidirectional

        self.fc_hidden_size = fc_hidden_size
        self.fc_layers = fc_layers
        self.dropout = dropout

        self.encoder = nn.LSTM(
            bidirectional=self.encoder_bidirectional,
            input_size=self.input_size,
            hidden_size=self.encoder_hidden_size,
            batch_first=True,
            num_layers=self.encoder_layers,
            dropout=(self.dropout if self.encoder_layers > 1 else 0)
        )
        self.decoder = nn.LSTM(
            bidirectional=self.decoder_bidirectional,
            input_size=(self.encoder_bidirectional + 1) * self.encoder_hidden_size,
            hidden_size=self.decoder_hidden_size,
            batch_first=True,
            num_layers=self.decoder_layers,
            dropout=(self.dropout if self.decoder_layers > 1 else 0)
        )

        self.fc_bloc = nn.Sequential(
            nn.Linear(
                in_features=self.fc_hidden_size,
                out_features=self.fc_hidden_size),
            nn.Dropout(self.dropout),
            nn.ReLU()
        )
        if self.fc_layers is not None:
            self.fc = nn.Sequential(
                nn.Linear(
                    in_features=(
                        (self.decoder_bidirectional + 1)
                        * self.decoder_hidden_size
                        * self.seq_length),
                    out_features=self.fc_hidden_size),
                nn.Dropout(self.dropout),
                nn.ReLU(),
                *[self.fc_bloc] * self.fc_layers,
                nn.Linear(
                    in_features=self.fc_hidden_size,
                    out_features=self.steps * self.output_size)
            )
        else:
            self.fc = nn.Linear(
                in_features=(
                    (self.decoder_bidirectional + 1)
                    * self.decoder_hidden_size
                    * self.seq_length),
                out_features=self.steps * self.output_size)
        self.init_weights()

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight.data)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias.data, 0.0)
            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight.data)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias.data, 0.0)
            elif isinstance(layer, nn.LSTM):
                for param in layer.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.constant_(param.data, 0)

    def forward(self, x):
        x, _ = self.encoder(x)
        x, _ = self.decoder(x)

        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


class CNNLSTM(nn.Module):
    def __init__(self, input_size, seq_length, output_size=1, steps=1,
                 cnn_channels=(16, 16), kernel_size=(2, 2), 
                 num_layers=1, hidden_size=16, bidirectional=False,
                 fc_layers=None, fc_hidden_size=64, dropout=0, act=nn.ReLU):
        super().__init__()

        self.input_size = input_size
        self.seq_length = seq_length
        self.output_size = output_size
        self.steps = steps

        self.cnn_channels = cnn_channels
        self.kernel_size = kernel_size

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.fc_hidden_size = fc_hidden_size
        self.fc_layers = fc_layers
        self.dropout = dropout

        self.drop = nn.Dropout(self.dropout)
        self.act = act()
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=self.cnn_channels[0],
                kernel_size=self.kernel_size[0]),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=self.cnn_channels[0],
                out_channels=self.cnn_channels[1],
                kernel_size=self.kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        seq_length_1 = l_out(self.seq_length, kernel_size=1, stride=1)
        seq_length_2 = l_out(seq_length_1, kernel_size=2, stride=1)
        seq_length_out = l_out(seq_length_2, kernel_size=2)
        self.lstm = nn.LSTM(
            bidirectional=self.bidirectional,
            input_size=int(self.cnn_channels[-1]),
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=(self.dropout if self.num_layers > 1 else 0)
        )

        self.fc_bloc = nn.Sequential(
            nn.Linear(
                in_features=self.fc_hidden_size,
                out_features=self.fc_hidden_size),
            self.drop,
            self.act
        )
        if self.fc_layers is not None:
            self.fc = nn.Sequential(
                nn.Linear(
                    in_features=(
                        (self.bidirectional + 1)
                        * self.hidden_size
                        * seq_length_out),
                    out_features=self.fc_hidden_size),
                self.drop,
                self.act,
                *[self.fc_bloc] * self.fc_layers,
                nn.Linear(
                    in_features=self.fc_hidden_size,
                    out_features=self.steps * self.output_size)
            )
        else:
            self.fc = nn.Linear(
                in_features=(
                    (self.bidirectional + 1)
                    * self.hidden_size
                    * seq_length_out),
                out_features=self.steps * self.output_size)
        self.init_weights()

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight.data)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias.data, 0.01)
            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight.data)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias.data, 0.0)
            elif isinstance(layer, nn.LSTM):
                for param in layer.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.constant_(param.data, 0)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = self.drop(x)

        x, _ = self.lstm(x)

        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


class CNN(nn.Module):
    def __init__(self, input_size, seq_length, out_channels=(16,32,64), droprate=0):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=out_channels[0],
                kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=out_channels[0],
                out_channels=out_channels[1],
                kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=out_channels[1],
                out_channels=out_channels[2],
                kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=(
                    int((int((seq_length - 2) / 2) - 2) / 2)
                    * out_channels[-1]), 
                out_features=1),
            nn.Dropout(droprate)
        )
        self.init_weights()

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight.data)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias.data, 0.0)
            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight.data)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias.data, 0.0)
            elif isinstance(layer, nn.LSTM):
                for param in layer.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.constant_(param.data, 0)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x
