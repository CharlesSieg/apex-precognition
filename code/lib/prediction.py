import time

import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate

from lib import logger


class Prediction(object):
  def __init__(self, model, device_name):
    super().__init__()
    self.device_name = device_name
    self.model = model

  def create(self, dataset, batch_size):
    start_time = time.time()
    logger.log.info(f"Prediction started at {start_time}...")

    unshuffled_dataloader = DataLoader(
      dataset,
      collate_fn=lambda x: [y.to(self.device_name) for y in default_collate(x)],
      batch_size=batch_size,
      shuffle=False,
      drop_last=True
    )
    P, Y = self.__make_predictions_from_dataloader(self.model, unshuffled_dataloader)
    pdf = pd.DataFrame([P, Y], index=['predictions', 'actuals']).T

    prediction_time = time.time() - start_time
    logger.log.info("Prediction time: {}".format(prediction_time))

    return pdf

  #########################################################
  # PRIVATE METHODS
  #########################################################

  def __make_predictions_from_dataloader(self, model, dataloader):
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
