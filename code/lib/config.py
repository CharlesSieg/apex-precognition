class Config(object):
  def __init__(self):
    super().__init__()
    self.batch_size = 1024 # 16
    self.learning_rate = 0.01 # 4e-4 == 0.0004
    self.n_dnn_layers = 5 # Number of fully-connected hidden layers
    self.n_epochs = 10
    self.n_features = 1 # Number of features (Set to 1 since this is a univariate timeseries.)
    self.n_hidden = 50 # Number of nodes/neurons in each hidden layer
    self.n_outputs = 1 # Prediction window
    self.perform_training = True # False
    self.split = 0.8
    self.tw = 180 # 5040 # 180 # Training window, i.e. number of steps to look back at for prediction.
    # n_features: number of input features (1 for univariate forecasting)
    # n_hidden: number of neurons in each hidden layer
    # n_outputs: number of outputs to predict for each training example
    # n_deep_layers: number of hidden dense layers after the lstm layer
    # sequence_len: number of steps to look back at for prediction
    # dropout: float (0 < dropout < 1) dropout ratio between dense layers
