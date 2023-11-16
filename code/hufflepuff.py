from lib import Config, Device, Forecast, Prediction, TimeSeriesData, Training
from lib import logger, plot
from lib.models import GRU, LSTM

import torch


config = Config()
device = Device()

# Load and prepare data.
data = TimeSeriesData(config)
df, dataset, trainloader, testloader = data.prepare()
plot.plot_history(df)

# Create and train the model.
model = LSTM(config, device_name=device.name)
model_path = "./.models/lstm.model"
#model = GRU(config, device_name=device.name)
model = model.to(device.name)

if config.perform_training:
  training = Training(model, device.name)
  t_losses, v_losses = training.start(config, trainloader, testloader)
  plot.plot_losses(t_losses, v_losses)
  import os
  if not os.path.exists("./.models"):
    os.makedirs("./.models")
  torch.save(model, "./.models/lstm.model")
  #region Make predictions.
  prediction = Prediction(model, device.name)
  predictions = prediction.create(dataset, config.batch_size)
  plot.plot_predictions(predictions)
  #endregion
else:
  model = torch.load(model_path)

# Create the forecast.
forecast = Forecast(model, df, "Close", config.tw)
f = forecast.create()
plot.plot_forecast(f)
