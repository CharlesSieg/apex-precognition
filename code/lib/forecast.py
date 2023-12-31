import time

import numpy as np
import pandas as pd
import torch

from lib import logger


class Forecast(object):
  def __init__(self, model, data, target, tw):
    super().__init__()
    self.data = data
    self.model = model
    self.target = target
    self.tw = tw

  def create(self):
    start_time = time.time()
    logger.log.info(f"Forecast started at {start_time}...")

    number_of_hours = 60
    number_of_steps = 60 * number_of_hours
    forecast = self.__n_step_forecast(number_of_steps)

    forecast_time = time.time() - start_time
    logger.log.info("Forecast time: {}".format(forecast_time))

    return forecast

  #########################################################
  # PRIVATE METHODS
  #########################################################

  def __one_step_forecast(self, history):
    '''
    history: a sequence of values representing the latest values of the time
    series, requirement -> len(history.shape) == 2

    outputs a single value which is the prediction of the next value in the
    sequence.
    '''
    # self.model.cpu()
    self.model.eval()
    with torch.no_grad():
      pre = torch.Tensor(history).unsqueeze(0)
      pred = self.model(pre)
    pred = pred.detach().cpu()
    return pred.numpy().reshape(-1)

  def __n_step_forecast(self, n: int, forecast_from: int=None):
    '''
    n: integer defining how many steps to forecast
    forecast_from: integer defining which index to forecast from. None if
    you want to forecast from the end.
    plot: True if you want to output a plot of the forecast, False if not.
    '''
    history = self.data[self.target].to_frame()

    # Create initial sequence input based on where in the series to forecast from.
    #if forecast_from:
    #  pre = list(history[forecast_from - self.tw : forecast_from][self.target].values)
    #else:
    pre = list(history[self.target])[-self.tw:]

    # Call one_step_forecast n times and append prediction to history
    for i, step in enumerate(range(n)):
      pre_ = np.array(pre[-self.tw:]).reshape(-1, 1)
      forecast = self.__one_step_forecast(pre_).squeeze()
      pre.append(forecast)

    res = history.copy()
    ls = [np.nan for i in range(len(history))]

    # Note: I have not handled the edge case where the start index + n crosses the end of the dataset.
    if forecast_from:
      ls[forecast_from : forecast_from + n] = list(np.array(pre[-n:]))
      res['forecast'] = ls
      res.columns = ['actual', 'forecast']
    else:
      fc = ls + list(np.array(pre[-n:]))
      ls = ls + [np.nan for i in range(len(pre[-n:]))]
      ls[:len(history)] = history[self.target].values
      res = pd.DataFrame([ls, fc], index=['actual', 'forecast']).T
      #forecast.index = pd.date_range('2017-07-01', periods=n, freq='D')

    return res
