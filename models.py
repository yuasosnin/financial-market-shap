import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm.notebook import tqdm


class BasicLSTM(nn.Module):
    def __init__(self, input_size, seq_length, hidden_size=128, num_layers=1):
        super().__init__()

        self.input_size = input_size
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.hidden_size*self.seq_length, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=1)
        )

    def forward(self, x, y):
        batch_size = x.shape[0]
        x = torch.cat((x, y), dim=2)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()

        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        return x
    
    
class ModelTrainer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.sceduler = None
        self.metric = lambda x, y: mean_squared_error(x, y) * 1e5
        self.logs = {}
        
    def forward(self, x, y):
        return self.model.forward(x, y)

    def log(self, name, value, how='array'):
#         if name not in self.logs:
#             self.logs[name] = np.array([])
#         if isinstance(value, float) or isinstance(value, int):
#             self.logs[name] = np.append(self.logs[name], value)
#         elif isinstance(value, np.ndarray):
#             self.logs[name] = np.concatenate(self.logs[name], value)
#         elif isinstance(value, torch.Tensor):
#             self.logs[name] = np.append(self.logs[name], value)
        if how == 'list':
            if name not in self.logs:
                self.logs[name] = list()
            self.logs[name].append(value)
        elif how == 'array':
            if name not in self.logs:
                self.logs[name] = np.array([])
            if isinstance(value, float) or isinstance(value, int):
                self.logs[name] = np.append(self.logs[name], value)
            elif isinstance(value, np.ndarray):
                self.logs[name] = np.concatenate(self.logs[name], value)
    
    @staticmethod
    def epoch_mean(array, loader_len):
        return array[-loader_len:].sum() / loader_len
    
    def training_step(self, batch):
        X, y, target  = batch
        X = X.to(self.device)
        y = y.to(self.device)
        target = target.to(self.device)
        output, x_attention, _ = self.forward(X, y)
        loss = self.criterion(output, target)

        self.log('train_step_loss', loss.item())
        mtr = self.metric(target.to('cpu'), output.detach().to('cpu'))
        self.log('train_step_metric', float(self.metric(target.to('cpu'), output.detach().to('cpu'))))
        return loss
    
    def validation_step(self, batch):
        X, y, target  = batch
        X = X.to(self.device)
        y = y.to(self.device)
        target = target.to(self.device)
        output, _, _ = self.forward(X, y)
        loss = self.criterion(output, target)
        
        self.log('valid_step_loss', loss.item())
        self.log('valid_step_metric', float(self.metric(target, output.detach())))
        return loss
    
    def prediction_step(self, batch):
        X, y, _ = batch
        X = X.to(self.device)
        y = y.to(self.device)
        output, x_attention, t_attention = self.forward(X, y)
        return output.to('cpu'), x_attention.to('cpu'), t_attention.to('cpu')
    
    def fit(self, train_loader, val_loader, epochs=10, verbose=True, stopper=None):
        if verbose: tqdm_ = tqdm
        else: tqdm_ = lambda x: x
        for epoch in tqdm_(range(epochs)):
            if verbose: print(f'{epoch+1} epoch:')

            self.train()
            for batch in tqdm_(train_loader):
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if self.sceduler is not None:
                self.sceduler.step()
            self.log(
                'train_epoch_loss', 
                self.epoch_mean(self.logs['train_step_loss'], len(train_loader)))
            self.log(
                'train_epoch_metric', 
                self.epoch_mean(self.logs['train_step_metric'], len(train_loader)))
        
            if verbose: 
                print(f"train loss {self.logs['train_epoch_loss'][-1]}")
                print(f"train {self.metric.__name__} {self.logs['train_epoch_metric'][-1]}")

            self.eval()
            for batch in tqdm_(val_loader):
                loss = self.validation_step(batch)
            self.log(
                'valid_epoch_loss', 
                self.logs['valid_step_loss'][-len(val_loader):].sum() / len(val_loader))
            self.log(
                'valid_epoch_metric', 
                self.logs['valid_step_metric'][-len(val_loader):].sum() / len(val_loader))
        
            if verbose: 
                print(f"valid loss {self.logs['valid_epoch_loss'][-1]}")
                print(f"train {self.metric.__name__} {self.logs['valid_epoch_metric'][-1]}")
            if verbose: print('-'*80)

    def predict(self, data_loader):
        pred = None
        x_attentions = []
        t_attentions = []
        self.eval()
        for batch in tqdm(data_loader):
            output, x_attention, t_attention = self.prediction_step(batch)
            #pred = np.concatenate((pred, output.detach().numpy()), 0)
            if pred is not None:
                pred = np.concatenate((pred, output.detach().numpy()))
            else: pred = output.detach().numpy()
            x_attentions.append(x_attention.detach())
            t_attentions.append(t_attention.detach())
        return pred, torch.cat(x_attentions, dim=0).numpy(), torch.cat(t_attentions, dim=0).numpy()

    def plot_loss(self):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('train loss')
        ax1.plot(self.logs['train_epoch_loss'])

        ax2 = ax1.twinx()
        ax2.set_ylabel('valid loss')
        ax2.plot(self.logs['valid_epoch_loss'], color='orange')

        fig.tight_layout()
        return fig
    
    def plot(self, x):
        return plt.plot(self.logs[x])
    
    
### alaxey kurochkin
class InputAttentionEncoder(nn.Module):
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
    
class DARNN_(nn.Module):
    def __init__(self, N, M, P, T, stateful_encoder=False, stateful_decoder=False):
        super(self.__class__, self).__init__()
        self.encoder = InputAttentionEncoder(N, M, T, stateful_encoder)
        self.decoder = TemporalAttentionDecoder(M, P, T, stateful_decoder)
    def forward(self, X_history, y_history):
        encoded_x, x_attention = self.encoder(X_history)
        out_y, t_attention = self.decoder(encoded_x, y_history)
        return out_y, x_attention, t_attention
    