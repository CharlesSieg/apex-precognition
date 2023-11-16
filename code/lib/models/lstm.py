import torch
import torch.nn as nn


class LSTM(nn.Module):
  def __init__(self, config, device_name, dropout=0.2):
    '''
    n_features: number of input features (1 for univariate forecasting)
    n_hidden: number of neurons in each hidden layer
    n_outputs: number of outputs to predict for each training example
    n_deep_layers: number of hidden dense layers after the lstm layer
    sequence_len: number of steps to look back at for prediction
    dropout: float (0 < dropout < 1) dropout ratio between dense layers
    '''
    super().__init__()
    self.device_name = device_name
    self.n_lstm_layers = 1 # config.n_lstm_layers
    self.n_hidden = config.n_hidden

    # LSTM Layer
    self.lstm = nn.LSTM(config.n_features, self.n_hidden, num_layers=self.n_lstm_layers, batch_first=True)

    # first dense after lstm
    self.fc1 = nn.Linear(config.n_hidden * config.tw, self.n_hidden)

    # Dropout layer
    self.dropout = nn.Dropout(p=dropout)

    # Create fully connected layers (n_hidden x n_dnn_layers)
    dnn_layers = []
    for i in range(config.n_dnn_layers):
      # Last layer (n_hidden x n_outputs)
      if i == config.n_dnn_layers - 1:
        dnn_layers.append(nn.ReLU())
        dnn_layers.append(nn.Linear(self.n_hidden, config.n_outputs))
      # All other layers (n_hidden x n_hidden) with dropout option
      else:
        dnn_layers.append(nn.ReLU())
        dnn_layers.append(nn.Linear(self.n_hidden, self.n_hidden))
        if dropout:
          dnn_layers.append(nn.Dropout(p=dropout))
    # compile DNN layers
    self.dnn = nn.Sequential(*dnn_layers)

  def forward(self, x):
    # Initialize hidden state
    hidden_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.n_hidden)
    cell_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.n_hidden)

    # move hidden state to device
    if self.device_name != "cpu":
      hidden_state = hidden_state.to(self.device_name)
      cell_state = cell_state.to(self.device_name)
      x = x.to(self.device_name)

    self.hidden = (hidden_state, cell_state)

    # Forward Pass
    x, h = self.lstm(x, self.hidden) # LSTM
    x = self.dropout(x.contiguous().view(x.shape[0], -1)) # Flatten lstm out
    x = self.fc1(x) # First Dense
    return self.dnn(x) # Pass forward through fully connected DNN.
