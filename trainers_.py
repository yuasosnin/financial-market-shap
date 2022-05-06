class ModelTrainer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.sceduler = None
        self.metric = lambda x, y: mean_squared_error(x, y) * 1e3
        self.logs = {}
        
    def forward(self, x, y):
        return self.model.forward(x, y)

    def log(self, name, value):
        if name not in self.logs:
            self.logs[name] = np.array([])
        if isinstance(value, float) or isinstance(value, int):
            self.logs[name] = np.append(self.logs[name], value)
        elif isinstance(value, np.ndarray):
            self.logs[name] = np.concatenate((self.logs[name], value))
    
    @staticmethod
    def epoch_mean(array, loader_len):
        return array[-loader_len:].sum() / loader_len

    def prepare_batch(self, batch):
        X, y, target  = batch
        X = X.to(self.device)
        y = y.to(self.device)
        target = target.to(self.device)
        return X, y, target
    
    def prepare_output(self, output):
        return output
    
    def training_step(self, batch):
        X, y, target  = self.prepare_batch(batch)
        output = self.forward(X, y)
        output = self.prepare_output(output)
        loss = self.criterion(output, target)

        self.log('train_step_loss', loss.item())
        self.log('train_step_metric', float(self.metric(target.flatten().cpu(), output.detach().flatten().cpu())))
        # self.log('train_y_true', target.cpu().numpy().flatten())
        # self.log('train_y_pred', output.detatch().cpu().numpy().flatten())
        return loss
    
    def validation_step(self, batch):
        X, y, target  = self.prepare_batch(batch)
        output = self.forward(X, y)
        output = self.prepare_output(output)
        loss = self.criterion(output, target)
        
        self.log('valid_step_loss', loss.item())
        self.log('valid_step_metric', float(self.metric(target.flatten(), output.detach().flatten())))
        # self.log('valid_y_true', target.cpu().numpy().flatten())
        # self.log('valid_y_pred', output.detatch().cpu().numpy().flatten())
        return loss
    
    def prediction_step(self, batch):
        X, y, _ = self.prepare_batch(batch)
        # if self.return_attention:
        #     output, x_attention, t_attention = self.forward(X, y)
        #     return output.to('cpu'), x_attention.to('cpu'), t_attention.to('cpu')
        # else:
        output = self.forward(X, y)
        return output.cpu()
    
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
                
            if val_loader is not None:
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
                    print(f"valid {self.metric.__name__} {self.logs['valid_epoch_metric'][-1]}")
            if verbose: print('-'*80)
    
    def predict(self, data_loader, verbose=True):
        if verbose: tqdm_ = tqdm
        else: tqdm_ = lambda x: x
        ret = {
            'y_true': np.array([]),
            'y_pred': np.array([]),
            'x_attentions': [],
            't_attentions': []
        }
        self.eval()
        for batch in tqdm_(data_loader):
            X, y, target  = batch
            x_attention, t_attention = None, None
            output = self.prediction_step(batch)
            if isinstance(output, tuple) and len(output)==3:
                output, x_attention, t_attention = output
            ret['y_pred'] = np.concatenate((ret['y_pred'], output.detach().numpy().flatten()))
            ret['y_true'] = np.concatenate((ret['y_true'], target.cpu().numpy().flatten()))
            if x_attention is not None:
                ret['x_attentions'].append(x_attention.detach())
                ret['t_attentions'].append(t_attention.detach())
        if x_attention is not None:
            ret['x_attentions'] = torch.cat(ret['x_attentions'])
            ret['t_attentions'] = torch.cat(ret['t_attentions'])
        return ret

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
    
    
# x = torch.cat((x, y), dim=2)
class TransformerTrainer(ModelTrainer):
    def __init__(self, model):
        super().__init__(model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.sceduler = None
        self.metric = lambda x, y: mean_squared_error(x, y) * 1e3
        self.logs = {}
        
    def forward(self, x):
        return self.model.forward(x)
    
    def prepare_batch(self, batch):
        X, y, target  = batch
        X = X.to(self.device)
        y = y.to(self.device)
        target = target.to(self.device)
        target = torch.cat([y[:,1:,:].flatten(1), target], dim=1)
        return X, y, target
    
    def training_step(self, batch):
        X, y, target  = self.prepare_batch(batch)
        X = torch.cat((X, y), dim=2)
        output = self.forward(X)
        output = self.prepare_output(output)
        loss = self.criterion(output, target)

        self.log('train_step_loss', loss.item())
        self.log('train_step_metric', float(self.metric(target.flatten().cpu(), output.detach().flatten().cpu())))
        # self.log('train_y_true', target.cpu().numpy().flatten())
        # self.log('train_y_pred', output.detatch().cpu().numpy().flatten())
        return loss
    
    def validation_step(self, batch):
        X, y, target  = self.prepare_batch(batch)
        X = torch.cat((X, y), dim=2)
        output = self.forward(X)
        output = self.prepare_output(output)
        loss = self.criterion(output, target)
        
        self.log('valid_step_loss', loss.item())
        self.log('valid_step_metric', float(self.metric(target.flatten(), output.detach().flatten())))
        # self.log('valid_y_true', target.cpu().numpy().flatten())
        # self.log('valid_y_pred', output.detatch().cpu().numpy().flatten())
        return loss

    
class AttentionTrainer(ModelTrainer):
    def __init__(self, model):
        super().__init__(model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.sceduler = None
        self.metric = lambda x, y: mean_squared_error(x, y) * 1e3
        self.logs = {}
    
    def prepare_output(self, output):
        output, _, _ = output
        return output
    
    def prediction_step(self, batch):
        X, y, _ = self.prepare_batch(batch)
        output, x_attention, t_attention = self.forward(X, y)
        return output.cpu(), x_attention.cpu(), t_attention.cpu()