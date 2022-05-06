import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm.notebook import tqdm
import math


################################################################################ baselines
class PreviousValueBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return y[:,-1,:]

class RollingMeanBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return y.mean(dim=1)


################################################################################ lstm
class VanillaLSTM(nn.Module):
    def __init__(self, input_size, seq_length, output_size=1, lstm_hidden_size=128, fc_hidden_size=64, dropout=0, lstm_layers=1, fc_layers=0, bidirectional=False, steps=1):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.seq_length = seq_length
        self.num_layers = lstm_layers
        self.bidirectional = bidirectional
        self.hidden_size = lstm_hidden_size
        self.fc_hidden_size = fc_hidden_size
        self.fc_layers = fc_layers
        self.dropout = dropout
        self.steps = steps

        self.lstm = nn.LSTM(
            bidirectional=self.bidirectional,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=(self.dropout if self.num_layers>1 else 0)
        )
        self.fc_bloc = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.Dropout(self.dropout),
            nn.ReLU()
        )
        # (self.bidirectional+1)*(self.num_layers)*hidden_size
        if self.fc_layers is not None:
            self.fc = nn.Sequential(
                nn.Linear(in_features=(self.bidirectional+1)*self.hidden_size*self.seq_length, out_features=self.hidden_size),
                nn.Dropout(self.dropout),
                nn.ReLU(),
                *[self.fc_bloc]*self.fc_layers,
                nn.Linear(in_features=self.hidden_size, out_features=self.steps*self.output_size)
            )
        else:
            self.fc = nn.Linear(in_features=(self.bidirectional+1)*self.hidden_size*self.seq_length, out_features=self.steps*self.output_size)

    def forward(self, x, y):
        batch_size = x.shape[0]
        x = torch.cat((x, y), dim=2)

        h0 = torch.zeros((self.bidirectional+1)*self.num_layers, batch_size, self.hidden_size).requires_grad_()
        c0 = torch.zeros((self.bidirectional+1)*self.num_layers, batch_size, self.hidden_size).requires_grad_()

        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = x.reshape(batch_size, -1)
        # print(x.shape)
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
        
        self.encoder_layers, self.decoder_layers = encoder_layers, decoder_layers
        self.encoder_hidden_size, self.decoder_hidden_size = encoder_hidden_size, decoder_hidden_size
        self.encoder_bidirectional, self.decoder_bidirectional = encoder_bidirectional, decoder_bidirectional
        
        self.fc_hidden_size = fc_hidden_size
        self.fc_layers = fc_layers
        self.dropout = dropout

        
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

    def forward(self, x, y):
        batch_size = x.shape[0]
        x = torch.cat((x, y), dim=2)

        h0e = torch.zeros((self.encoder_bidirectional+1)*self.encoder_layers, batch_size, self.encoder_hidden_size).requires_grad_()
        c0e = torch.zeros((self.encoder_bidirectional+1)*self.encoder_layers, batch_size, self.encoder_hidden_size).requires_grad_()
        
        h0d = torch.zeros((self.decoder_bidirectional+1)*self.decoder_layers, batch_size, self.decoder_hidden_size).requires_grad_()
        c0d = torch.zeros((self.decoder_bidirectional+1)*self.decoder_layers, batch_size, self.decoder_hidden_size).requires_grad_()

        x, (hn, cn) = self.encoder(x, (h0e, c0e))
        x, (hn, cn) = self.decoder(x, (h0d, c0d))
        x = x.reshape(batch_size, -1)
        # print(x.shape)
        x = self.fc(x)
        return x


################################################################################ transformer
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
        
        # self.src_mask = None
        self.pos_encoder = PositionalEncoding(self.input_size)
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
    
    
################################################################################ darnn
class InputAttentionEncoder(nn.Module): ### alaxey kurochkin
    def __init__(self, N, M, T, stateful=False):
        """
        :param: N: int
            number of time serieses
        :param: M:
            number of LSTM units
        :param: T:
            number of timesteps
        :param: stateful:
            decides whether to initialize cell state of new time window with values of the last cell state
            of previous time window or to initialize it with zeros
        """
        super(self.__class__, self).__init__()
        self.N = N
        self.M = M
        self.T = T
        
        self.encoder_lstm = nn.LSTMCell(input_size=self.N, hidden_size=self.M)
        
        #equation 8 matrices
        
        self.W_e = nn.Linear(2*self.M, self.T)
        self.U_e = nn.Linear(self.T, self.T, bias=False)
        self.v_e = nn.Linear(self.T, 1, bias=False)
    
    def forward(self, inputs):
        encoded_inputs = torch.zeros((inputs.size(0), self.T, self.M))
        attentions = torch.zeros((inputs.size(0), self.T, self.N))
        
        #initiale hidden states
        h_tm1 = torch.zeros((inputs.size(0), self.M))
        s_tm1 = torch.zeros((inputs.size(0), self.M))
        
        for t in range(self.T):
            #concatenate hidden states
            h_c_concat = torch.cat((h_tm1, s_tm1), dim=1)
            
            #attention weights for each k in N (equation 8)
            x = self.W_e(h_c_concat).unsqueeze_(1).repeat(1, self.N, 1)
            y = self.U_e(inputs.permute(0, 2, 1))
            z = torch.tanh(x + y)
            e_k_t = torch.squeeze(self.v_e(z))
        
            #normalize attention weights (equation 9)
            alpha_k_t = F.softmax(e_k_t, dim=1)
                
            #weight inputs (equation 10)
            weighted_inputs = alpha_k_t * inputs[:, t, :] 
    
            #calculate next hidden states (equation 11)
            h_tm1, s_tm1 = self.encoder_lstm(weighted_inputs, (h_tm1, s_tm1))
            
            encoded_inputs[:, t, :] = h_tm1
            attentions[:, t, :] = alpha_k_t
        return encoded_inputs, attentions
    
    
class TemporalAttentionDecoder(nn.Module):
    def __init__(self, M, P, T, stateful=False):
        """
        :param: M: int
            number of encoder LSTM units
        :param: P:
            number of deocder LSTM units
        :param: T:
            number of timesteps
        :param: stateful:
            decides whether to initialize cell state of new time window with values of the last cell state
            of previous time window or to initialize it with zeros
        """
        super(self.__class__, self).__init__()
        self.M = M
        self.P = P
        self.T = T
        self.stateful = stateful
        
        self.decoder_lstm = nn.LSTMCell(input_size=1, hidden_size=self.P)
        
        #equation 12 matrices
        self.W_d = nn.Linear(2*self.P, self.M)
        self.U_d = nn.Linear(self.M, self.M, bias=False)
        self.v_d = nn.Linear(self.M, 1, bias = False)
        
        #equation 15 matrix
        self.w_tilda = nn.Linear(self.M + 1, 1)
        
        #equation 22 matrices
        self.W_y = nn.Linear(self.P + self.M, self.P)
        self.v_y = nn.Linear(self.P, 1)
        
    def forward(self, encoded_inputs, y):
        
        #initializing hidden states
        d_tm1 = torch.zeros((encoded_inputs.size(0), self.P))
        s_prime_tm1 = torch.zeros((encoded_inputs.size(0), self.P))
                
        attentions = torch.zeros((encoded_inputs.size(0), self.T, self.T))
                
        for t in range(self.T):
            #concatenate hidden states
            d_s_prime_concat = torch.cat((d_tm1, s_prime_tm1), dim=1)
            
            #temporal attention weights (equation 12)
            x1 = self.W_d(d_s_prime_concat).unsqueeze_(1).repeat(1, encoded_inputs.shape[1], 1)
            y1 = self.U_d(encoded_inputs)
            z1 = torch.tanh(x1 + y1)
            l_i_t = self.v_d(z1)
            
            #normalized attention weights (equation 13)
            beta_i_t = F.softmax(l_i_t, dim=1)
            attentions[:, t, :] = beta_i_t.view(-1, self.T)
            #print(beta_i_t.shape)
            
            #create context vector (equation_14)
            c_t = torch.sum(beta_i_t * encoded_inputs, dim=1)
            
            #concatenate c_t and y_t
            y_c_concat = torch.cat((c_t, y[:, t, :]), dim=1)
            #create y_tilda
            y_tilda_t = self.w_tilda(y_c_concat)
            
            #calculate next hidden states (equation 16)
            d_tm1, s_prime_tm1 = self.decoder_lstm(y_tilda_t, (d_tm1, s_prime_tm1))
        
        #concatenate context vector at step T and hidden state at step T
        d_c_concat = torch.cat((d_tm1, c_t), dim=1)

        #calculate output
        y_Tp1 = self.v_y(self.W_y(d_c_concat))
        return y_Tp1, attentions

    
class DARNN(nn.Module):
    def __init__(self, N, M, P, T, stateful_encoder=False, stateful_decoder=False):
        super(self.__class__, self).__init__()
        self.encoder = InputAttentionEncoder(N, M, T, stateful_encoder)
        self.decoder = TemporalAttentionDecoder(M, P, T, stateful_decoder)
    def forward(self, X_history, y_history):
        encoded_x, x_attention = self.encoder(X_history)
        out_y, t_attention = self.decoder(encoded_x, y_history)
        return out_y, x_attention, t_attention
    
