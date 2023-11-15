import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
  def __init__(self, df, target, features, sequence_length=5):
    self.df = df
    self.features = features
    self.target = target
    self.sequence_length = sequence_length

    #self.X = torch.tensor(df[features].values).float()
    #self.y = torch.tensor(df[target].values).float()

  def __getitem__(self, idx):
    sample = self.df[idx]
    return torch.Tensor(sample['sequence']), torch.Tensor(sample['target'])
    # if i >= self.sequence_length - 1:
    #   i_start = i - self.sequence_length + 1
    #   x = self.X[i_start:(i + 1), :]
    # else:
    #   padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
    #   x = self.X[0:(i + 1), :]
    #   x = torch.cat((padding, x), 0)
    # return x, self.y[i]

  def __len__(self):
    return len(self.df)
    #return self.X.shape[0]
