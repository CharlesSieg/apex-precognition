#region Imports

import time

from lib import Device, Forecast, Prediction, TimeSeriesData, Training
from lib import logger, plot
from lib.models import GRU, LSTM

import torch

#endregion

logger.log_manager.configure("DEBUG", "./.logs", "foo")
logger.log.info(logger.LINE_BREAK)
device = Device()

BATCH_SIZE = 1024 # 16
learning_rate = 0.01 # 4e-4 == 0.0004
n_dnn_layers = 5 # Number of fully-connected hidden layers
n_epochs = 10
n_features = 1 # Number of features (Set to 1 since this is a univariate timeseries.)
n_hidden = 50 # Number of nodes/neurons in each hidden layer
n_outputs = 1 # Prediction window
perform_training = False
split = 0.8
tw = 180 # 5040 # 180 # Training window, i.e. number of steps to look back at for prediction.
# n_features: number of input features (1 for univariate forecasting)
# n_hidden: number of neurons in each hidden layer
# n_outputs: number of outputs to predict for each training example
# n_deep_layers: number of hidden dense layers after the lstm layer
# sequence_len: number of steps to look back at for prediction
# dropout: float (0 < dropout < 1) dropout ratio between dense layers

# Load and prepare data.
data = TimeSeriesData(BATCH_SIZE, n_outputs, split, tw)
df, trainloader, testloader = data.prepare()
plot.plot_history(df)

# Create and train the model.
model = LSTM(n_features, n_hidden, n_outputs, tw, n_deep_layers=n_dnn_layers, device_name=device.name)
model_path = "./.models/lstm.model"
#model = GRU(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, device_name=device.name)
model = model.to(device.name)

if perform_training:
  training = Training(model, device.name)
  t_losses, v_losses = training.start(n_epochs, learning_rate, trainloader, testloader)
  plot.plot_losses(t_losses, v_losses)
  torch.save(model, "./.models/lstm.model")
  # Make predictions.
  prediction = Prediction(model, device.name)
  predictions = prediction.create(dataset, BATCH_SIZE)
  plot.plot_predictions(predictions)
else:
  model = torch.load(model_path)

# Create the forecast.
forecast = Forecast(model, df, "Close", tw)
f = forecast.create()
plot.plot_forecast(f)
