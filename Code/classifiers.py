import numpy as np
import pandas as pd


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MyNN(nn.Module):
    def __init__(self, n_inputs=1, n_hidden=10, dropout=0.2):
        super(MyNN, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.act1 = nn.Tanh()
        self.dout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(n_hidden, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        if self.dropout > 0:
            x = self.dout(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x
    
    
    
    
from sklearn.preprocessing import StandardScaler
class FitMyNN(object):
    
    def __init__(self, n_hidden=10, dropout=0.2, n_epochs=10, batch_size=64, lr=0.01, lam=0., optimizer='Adam', debug=0):        
        self.model = None
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lam = lam
        self.optimizer = optimizer
        self.debug = debug
        self.scaler = StandardScaler()
        
    
    def fit(self, X, y):
        # Scaling
        X_ss = self.scaler.fit_transform(X)
        # Estimate model
        self.model = MyNN(n_inputs=X_ss.shape[1], n_hidden=self.n_hidden, dropout=self.dropout)
        # Convert X and y into torch tensors
        X_tensor = torch.as_tensor(X_ss, dtype=torch.float32, device=device)
        y_tensor = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)
        # Create dataset for trainig procedure
        train_data = TensorDataset(X_tensor, y_tensor)
        # Estimate loss
        loss_func = nn.BCELoss()
        # Estimate optimizer
        if self.optimizer == "Adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "RMSprop":
            opt = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Enable droout
        self.model.train(True)
        # Start the model fit
        for epoch_i in range(self.n_epochs):
            loss_history = []
            for x_batch, y_batch in DataLoader(train_data, batch_size=self.batch_size, shuffle=True):
                # make prediction on a batch
                y_pred_batch = self.model(x_batch)
                # calculate loss
                loss = loss_func(y_pred_batch, y_batch)
                #loss = -(torch.log(y_pred_batch) * y_batch).sum(-1).mean()
                lam = torch.tensor(self.lam)
                l2_reg = torch.tensor(0.)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)
                loss += lam * l2_reg
                # set gradients to zero
                opt.zero_grad()
                # backpropagate gradients
                loss.backward()
                # update the model weights
                opt.step()
                loss_history.append(loss.item())
            if self.debug:
                print("epoch: %i, mean loss: %.5f" % (epoch_i, np.mean(loss_history)))
    
    def predict_proba(self, X):
        # Scaling
        X_ss = self.scaler.transform(X)
        # Disable droout
        self.model.train(False)
        # Convert X and y into torch tensors
        X_tensor = torch.as_tensor(X_ss, dtype=torch.float32, device=device)
        # Make predictions for X 
        y_pred = self.model(X_tensor)
        y_pred_1 = y_pred.cpu().detach().numpy()
        y_pred = np.hstack((1-y_pred_1, y_pred_1))
        return y_pred
    
    
    
    
# Regreesions
class MyNNRegressor(nn.Module):
    def __init__(self, n_inputs=1, n_hidden=10, dropout=0.2):
        super(MyNNRegressor, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.act1 = nn.Tanh()
        self.dout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        if self.dropout > 0:
            x = self.dout(x)
        x = self.fc2(x)
        return x
    
    
class FitMyNN_RuLSIF(object):
    
    def __init__(self, n_hidden=10, dropout=0.2, n_epochs=10, batch_size=64, lr=0.01, lam=0., alpha=0., optimizer='Adam', debug=0):        
        self.model = None
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lam = lam
        self.alpha = alpha
        self.optimizer = optimizer
        self.debug = debug
        self.scaler = StandardScaler()
        
    
    def fit(self, X, y):
        # Scaling
        X_ss = self.scaler.fit_transform(X)
        # Estimate model
        self.model = MyNNRegressor(n_inputs=X_ss.shape[1], n_hidden=self.n_hidden, dropout=self.dropout)
        # Convert X and y into torch tensors
        X_tensor = torch.as_tensor(X_ss, dtype=torch.float32, device=device)
        y_tensor = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)
        # Create dataset for trainig procedure
        train_data = TensorDataset(X_tensor, y_tensor)
        # Estimate loss
        loss_func = nn.MSELoss()
        # Estimate optimizer
        if self.optimizer == "Adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "RMSprop":
            opt = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Enable droout
        self.model.train(True)
        # Start the model fit
        for epoch_i in range(self.n_epochs):
            loss_history = []
            for x_batch, y_batch in DataLoader(train_data, batch_size=self.batch_size, shuffle=True):
                # make prediction on a batch
                y_pred_batch = self.model(x_batch)
                y_pred_batch_ref = y_pred_batch[y_batch == 0]
                y_pred_batch_test = y_pred_batch[y_batch == 1]
                loss = 0.5 * (1 - self.alpha) * (y_pred_batch_ref**2).mean() + \
                       0.5 *      self.alpha  * (y_pred_batch_test**2).mean() - (y_pred_batch_test).mean()
                lam = torch.tensor(self.lam)
                l2_reg = torch.tensor(0.)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)
                loss += lam * l2_reg
                # set gradients to zero
                opt.zero_grad()
                # backpropagate gradients
                loss.backward()
                # update the model weights
                opt.step()
                loss_history.append(loss.item())
            if self.debug:
                print("epoch: %i, mean loss: %.5f" % (epoch_i, np.mean(loss_history)))
    
    def predict_proba(self, X):
        # Scaling
        X_ss = self.scaler.transform(X)
        # Disable droout
        self.model.train(False)
        # Convert X and y into torch tensors
        X_tensor = torch.as_tensor(X_ss, dtype=torch.float32, device=device)
        # Make predictions for X 
        y_pred = self.model(X_tensor)
        y_pred = y_pred.cpu().detach().numpy()
        y_pred[y_pred <= 0] *= 0
        return y_pred
    
    

class FitMyNN_Exp(object):
    
    def __init__(self, n_hidden=10, dropout=0.2, n_epochs=10, batch_size=64, lr=0.01, lam=0., alpha=0., optimizer='Adam', debug=0):        
        self.model = None
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lam = lam
        self.alpha = alpha
        self.optimizer = optimizer
        self.debug = debug
        self.scaler = StandardScaler()
        
    
    def fit(self, X, y):
        # Scaling
        X_ss = self.scaler.fit_transform(X)
        # Estimate model
        self.model = MyNNRegressor(n_inputs=X_ss.shape[1], n_hidden=self.n_hidden, dropout=self.dropout)
        # Convert X and y into torch tensors
        X_tensor = torch.as_tensor(X_ss, dtype=torch.float32, device=device)
        y_tensor = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)
        # Create dataset for trainig procedure
        train_data = TensorDataset(X_tensor, y_tensor)
        # Estimate loss
        loss_func = nn.MSELoss()
        # Estimate optimizer
        if self.optimizer == "Adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "RMSprop":
            opt = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Enable droout
        self.model.train(True)
        # Start the model fit
        for epoch_i in range(self.n_epochs):
            loss_history = []
            for x_batch, y_batch in DataLoader(train_data, batch_size=self.batch_size, shuffle=True):
                # make prediction on a batch
                y_pred_batch = self.model(x_batch)
                y_pred_batch_ref = y_pred_batch[y_batch == 0]
                y_pred_batch_test = y_pred_batch[y_batch == 1]
                loss = torch.exp(-(2*y_batch - 1) * y_pred_batch).mean()
                lam = torch.tensor(self.lam)
                l2_reg = torch.tensor(0.)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)
                loss += lam * l2_reg
                #loss = -(torch.log(y_pred_batch) * y_batch).sum(-1).mean()
                # set gradients to zero
                opt.zero_grad()
                # backpropagate gradients
                loss.backward()
                # update the model weights
                opt.step()
                loss_history.append(loss.item())
            if self.debug:
                print("epoch: %i, mean loss: %.5f" % (epoch_i, np.mean(loss_history)))
    
    def predict_proba(self, X):
        # Scaling
        X_ss = self.scaler.transform(X)
        # Disable droout
        self.model.train(False)
        # Convert X and y into torch tensors
        X_tensor = torch.as_tensor(X_ss, dtype=torch.float32, device=device)
        # Make predictions for X 
        y_pred = self.model(X_tensor)
        y_pred = y_pred.cpu().detach().numpy()
        return y_pred
    
    
    
# RNN
class MyRNN(nn.Module):
    def __init__(self, n_inputs=1, n_hidden=10, dropout=0.2):
        super(MyRNN, self).__init__()
        self.dropout = dropout
        self.n_hidden = n_hidden
        self.rnn1 = nn.LSTM(n_inputs, n_hidden, 1, dropout=dropout)
        self.fc1 = nn.Linear(n_hidden, 1)
        self.ac1 = nn.Sigmoid()

    def forward(self, x):
        hidden = torch.zeros(1, x.size()[1], self.n_hidden)
        output, _ = self.rnn1(x)
        output = output[-1]
        output = self.fc1(output)
        output = self.ac1(output)
        return output
    
    
from sklearn.preprocessing import StandardScaler

class FitMyRNN(object):
    
    def __init__(self, n_hidden=10, dropout=0.2, n_epochs=10, batch_size=64, lr=0.01, lam=0., optimizer='Adam', debug=0):        
        self.model = None
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lam = lam
        self.optimizer = optimizer
        self.debug = debug
        self.scaler = StandardScaler()
        
    
    def fit(self, X, y):
        # Scaling
        self.scaler.fit(X[0, :, :])
        for i in range(X.shape[0]):
            X[i, :, :] = self.scaler.transform(X[i, :, :]) 
        # Estimate model
        self.model = MyRNN(n_inputs=X.shape[2], n_hidden=self.n_hidden, dropout=self.dropout)
        # Convert X and y into torch tensors
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device).permute(1, 2, 0)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=device)
        # Create dataset for trainig procedure
        train_data = TensorDataset(X_tensor, y_tensor)
        # Estimate loss
        loss_func = nn.BCELoss()
        # Estimate optimizer
        if self.optimizer == "Adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "RMSprop":
            opt = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Enable droout
        self.model.train(True)
        # Start the model fit
        for epoch_i in range(self.n_epochs):
            loss_history = []
            for x_batch, y_batch in DataLoader(train_data, batch_size=self.batch_size, shuffle=True):
                # make prediction on a batch
                x_batch = x_batch.permute(2, 0, 1)
                y_pred_batch = self.model(x_batch)#.view(-1)
                # calculate loss
                loss = loss_func(y_pred_batch.view(-1), y_batch.view(-1))
                lam = torch.tensor(self.lam)
                l2_reg = torch.tensor(0.)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)
                loss += lam * l2_reg
                # set gradients to zero
                opt.zero_grad()
                # backpropagate gradients
                loss.backward()
                # update the model weights
                opt.step()
                loss_history.append(loss.item())
            if self.debug:
                print("epoch: %i, mean loss: %.5f" % (epoch_i, np.mean(loss_history)))
    
    def predict_proba(self, X):
        # Scaling
        for i in range(X.shape[0]):
            X[i, :, :] = self.scaler.transform(X[i, :, :]) 
        # Disable droout
        self.model.train(False)
        # Convert X and y into torch tensors
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        # Make predictions for X 
        y_pred = self.model(X_tensor)
        y_pred_1 = y_pred.cpu().detach().numpy()
        y_pred = np.hstack((1-y_pred_1, y_pred_1))
        return y_pred
    
    
    

# GBDT-RuLSIF    
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


class GBDTRulSIF(object):
    
    def __init__(self, n_estimators=100, learning_rate=0.1, sample_frac=0.7, alpha=0.1, 
                 max_depth=4, min_samples_leaf=1, min_samples_split=2, splitter='best', 
                 max_features=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.sample_frac = sample_frac
        self.alpha = alpha
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.splitter = splitter
        self.max_features = max_features
        self.estimators = []
        
        
    def pe_loss(self, pred, y):
        L = (-0.5 *       self.alpha *  (pred)**2) * (y == 1) + \
            (-0.5 * (1. - self.alpha) * (pred)**2) * (y == 0) + pred * (y == 1) - 0.5
        return -L
    
    
    def pe_loss_grad(self, pred, y):
        dL = (-1. *       self.alpha  * (pred)) * (y == 1) + \
             (-1. * (1. - self.alpha) * (pred)) * (y == 0) + 1. * (y == 1)
        return -dL
    
    
    def fit(self, X, y):
        for i in range(self.n_estimators):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.sample_frac)
            pred_train = self.predict_proba(X_train)
            grad = - self.pe_loss_grad(pred_train, y_train)
            atree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, 
                                          min_samples_split=self.min_samples_split, splitter=self.splitter, 
                                          max_features=self.max_features)
            atree.fit(X_train, grad)
            self.estimators.append(atree)
        
        
    def predict_proba(self, X):
        predictions = 1. + 0.1 * np.random.rand(len(X))
        for est in self.estimators:
            pred = est.predict(X)
            predictions += self.learning_rate * pred
        return predictions