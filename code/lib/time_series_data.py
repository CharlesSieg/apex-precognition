import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from lib import SequenceDataset

from lib import logger


class TimeSeriesData(object):
  def __init__(self, config):
    super().__init__()
    self.batch_size = config.batch_size
    self.n_outputs = config.n_outputs
    self.training_split = config.split
    self.training_window = config.tw

  def prepare(self):
    # Load the stock pricing data.
    stock_data = self.__load_stock_data()

    # Using only the Close value for now.
    #data = stock_data.filter(["Date", "Close"])

    #data = data.set_index('Date').interpolate()
    stock_data.set_index("Date", inplace=True)
    df = stock_data.copy().dropna(axis=0)

    # from sklearn.preprocessing import StandardScaler
    # scalers = {}
    # for x in df.columns:
    #   scalers[x] = StandardScaler().fit(df[x].values.reshape(-1, 1))

    # norm_df = df.copy()
    # for i, key in enumerate(scalers.keys()):
    #   norm = scalers[key].transform(norm_df.iloc[:, i].values.reshape(-1, 1))
    #   norm_df.iloc[:, i] = norm

    sequences = self.__generate_sequences(df.Close.to_frame(), self.training_window, self.n_outputs, 'Close')
    dataset = SequenceDataset(sequences, target="Open", features=["Close", "Volume"])
    train_len = int(len(dataset) * self.training_split)
    lens = [train_len, len(dataset) - train_len]
    train_ds, test_ds = random_split(dataset, lens)
    # from torch.utils.data import Subset
    #train_ds = Subset(dataset, range(lens[0]))
    #test_ds = Subset(dataset, range(lens[0],lens[0]+lens[1]))
    torch.manual_seed(69)
    trainloader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
    return df, dataset, trainloader, testloader

  #########################################################
  # PRIVATE METHODS
  #########################################################

  def __generate_sequences(self, df: pd.DataFrame, tw: int, pw: int, target_columns):
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

  def __load_stock_data(self):
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
