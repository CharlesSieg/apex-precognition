#region Imports

import os
import time

from lib import Device, Forecast, SequenceDataset
from lib import logger, plot
from lib.models import GRU, LSTM

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate

#endregion

logger.log_manager.configure("DEBUG", "./.logs", "foo")
logger.log.info(logger.LINE_BREAK)
device = Device()

def generate_sequences(df: pd.DataFrame, tw: int, pw: int, target_columns):
  '''
  df: Pandas DataFrame of the univariate time-series
  tw: Training Window - Integer defining how many steps to look back
  pw: Prediction Window - Integer defining how many steps to predict

  returns: dictionary of sequences and targets for all sequences
  '''
  data = dict() # Store results into a dictionary
  L = len(df)
  for i in range(L-tw):
    # Get current sequence
    sequence = df[i:i+tw].values
    # Get values right after the current sequence
    target = df[i+tw:i+tw+pw][target_columns].values
    data[i] = {'sequence': sequence, 'target': target}
  return data

def load_stock_data():
  # Input data files are available in the read-only "../data/" directory
  for dirname, _, filenames in os.walk('../data'):
      for filename in filenames:
          logger.log.debug(os.path.join(dirname, filename))

  logger.log.info("Loading stock prices...")
  stock_data = pd.read_csv("../data/TSLA.csv")
  stock_data.head()

  # Combine Date and Time columns into single Date column in proper timestamp format.
  stock_data.loc[:, "Date"] = pd.to_datetime(stock_data.Date.astype(str) + ' ' + stock_data.Time.astype(str))
  stock_data = stock_data.drop("Time", axis=1)

  logger.log.debug(stock_data)
  logger.log.info(f"{len(stock_data)} prices loaded.")
  return stock_data

def make_predictions_from_dataloader(model, dataloader):
  model.eval()
  predictions, actuals = [], []
  for x, y in dataloader:
    with torch.no_grad():
      p = model(x)
      #p = p[:,-1,:]
      predictions.append(p)
      actuals.append(y.squeeze())
  predictions = torch.cat(predictions).cpu()
  predictions = predictions.numpy()
  actuals = torch.cat(actuals).cpu()
  actuals = actuals.numpy()
  return predictions.squeeze(), actuals



BATCH_SIZE = 1024 # 16
learning_rate = 0.01 # 4e-4 == 0.0004
n_dnn_layers = 5 # Number of fully-connected hidden layers
n_epochs = 10
n_features = 1 # Number of features (Set to 1 since this is a univariate timeseries.)
n_hidden = 50 # Number of nodes/neurons in each hidden layer
n_outputs = 1 # Prediction window
split = 0.8
tw = 180 # 5040 # 180 # Training window, i.e. number of steps to look back at for prediction.

    # n_features: number of input features (1 for univariate forecasting)
    # n_hidden: number of neurons in each hidden layer
    # n_outputs: number of outputs to predict for each training example
    # n_deep_layers: number of hidden dense layers after the lstm layer
    # sequence_len: number of steps to look back at for prediction
    # dropout: float (0 < dropout < 1) dropout ratio between dense layers

#region Data Preparation

# Load the stock pricing data.
stock_data = load_stock_data()

# Using only the Close value for now.
#data = stock_data.filter(["Date", "Close"])

#data = data.set_index('Date').interpolate()
stock_data.set_index("Date", inplace=True)
df = stock_data.copy().dropna(axis=0)

plot.plot_history(df)

# from sklearn.preprocessing import StandardScaler
# scalers = {}
# for x in df.columns:
#   scalers[x] = StandardScaler().fit(df[x].values.reshape(-1, 1))

# norm_df = df.copy()
# for i, key in enumerate(scalers.keys()):
#   norm = scalers[key].transform(norm_df.iloc[:, i].values.reshape(-1, 1))
#   norm_df.iloc[:, i] = norm

sequences = generate_sequences(df.Close.to_frame(), tw, n_outputs, 'Close')
dataset = SequenceDataset(sequences, target="Open", features=["Close", "Volume"])

train_len = int(len(dataset) * split)
lens = [train_len, len(dataset) - train_len]
train_ds, test_ds = random_split(dataset, lens)
# from torch.utils.data import Subset
#train_ds = Subset(dataset, range(lens[0]))
#test_ds = Subset(dataset, range(lens[0],lens[0]+lens[1]))
torch.manual_seed(69)
trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

#endregion

#region Train

start_time = time.time()
logger.log.info(f"Training started at {start_time}...")

model = LSTM(n_features, n_hidden, n_outputs, tw, n_deep_layers=n_dnn_layers, device_name=device.name).to(device.name)
#model = GRU(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, device_name=device.name).to(device.name)
criterion = nn.MSELoss(reduction="mean").to(device.name)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
t_losses, v_losses = [], []

for epoch in range(n_epochs):
  epoch_training_start_time = time.time()

  train_loss, valid_loss = 0.0, 0.0
  model.train()

  for x, y in trainloader:
    optimizer.zero_grad()
    x = x.to(device.name)
    y = y.to(device.name)

    preds = model(x)

    loss = criterion(preds, y)
    train_loss += loss.item()
    loss.backward()
    optimizer.step()

  epoch_loss = train_loss / len(trainloader)
  t_losses.append(epoch_loss)

  # validation step
  model.eval()
  for x, y in testloader:
    with torch.no_grad():
      x, y = x.to(device.name), y.squeeze().to(device.name)
      preds = model(x).squeeze()
      error = criterion(preds, y)
    valid_loss += error.item()
  valid_loss = valid_loss / len(testloader)
  v_losses.append(valid_loss)

  logger.log.info(f'{epoch} - train: {epoch_loss}, valid: {valid_loss}')
  epoch_training_time = time.time() - epoch_training_start_time
  logger.log.info("Epoch training time: {}".format(epoch_training_time))

training_time = time.time() - start_time
logger.log.info("Training time: {}".format(training_time))
plot.plot_losses(t_losses, v_losses)

#endregion

#region Predict

start_time = time.time()
logger.log.info(f"Prediction started at {start_time}...")
unshuffled_dataloader = DataLoader(dataset, collate_fn=lambda x: [y.to(device.name) for y in default_collate(x)], batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
P, Y = make_predictions_from_dataloader(model, unshuffled_dataloader)
P.shape, Y.shape

pdf = pd.DataFrame([P, Y], index=['predictions', 'actuals']).T
plot.plot_predictions(pdf)

prediction_time = time.time() - start_time
logger.log.info("Prediction time: {}".format(prediction_time))

#endregion

# Create and plot the forecast.
forecast = Forecast(model, df, 'Close', tw)
f = forecast.create()
plot.plot_forecast(f)
