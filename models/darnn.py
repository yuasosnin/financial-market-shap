import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm.notebook import tqdm
from .utils import *
# from .trainer import Trainer


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
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        self.encoder = InputAttentionEncoder(N, M, T, stateful_encoder)
        self.decoder = TemporalAttentionDecoder(M, P, T, stateful_decoder)
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
                        
    def forward(self, x, y):
        encoded_x, x_attention = self.encoder(x)
        out, t_attention = self.decoder(encoded_x, y)
        return out, x_attention, t_attention
   
    
# class DARNNTrainer(Trainer):       
#     def forward(self, x, y, return_attention=False):
#         output, x_attention, t_attention = self.model.forward(x, y)
#         if self.task == 'classification':
#             output = torch.sigmoid(output)
#         if return_attention:
#             return output, x_attention, t_attention
#         else:
#             return output
    
#     def prediction_step(self, batch):
#         x, y, _ = self._prepare_batch(batch)
#         output, x_attention, t_attention = self.forward(x, y, return_attention=True)
#         return output.cpu(), x_attention.cpu(), t_attention.cpu()

#     def predict(self, data_loader, verbose=True):
#         if verbose: tqdm_ = tqdm
#         else: tqdm_ = lambda x: x
#         ret = {
#             'y_true': np.array([]),
#             'y_pred': np.array([]),
#             'x_attentions': [],
#             't_attentions': []
#         }
#         self.eval()
#         for batch in tqdm_(data_loader):
#             X, y, target  = batch
#             x_attention, t_attention = None, None
#             output = self.prediction_step(batch)
#             if isinstance(output, tuple) and len(output)==3:
#                 output, x_attention, t_attention = output
#             ret['y_pred'] = np.concatenate((ret['y_pred'], output.detach().numpy().flatten()))
#             ret['y_true'] = np.concatenate((ret['y_true'], target.cpu().numpy().flatten()))
#             if x_attention is not None:
#                 ret['x_attentions'].append(x_attention.detach())
#                 ret['t_attentions'].append(t_attention.detach())
#         if x_attention is not None:
#             ret['x_attentions'] = torch.cat(ret['x_attentions'])
#             ret['t_attentions'] = torch.cat(ret['t_attentions'])
#         return ret