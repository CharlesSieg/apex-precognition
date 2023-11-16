import time

import torch
import torch.nn as nn

from lib import logger


class Training(object):
  def __init__(self, model, device_name):
    super().__init__()
    self.device_name = device_name
    self.model = model

  def start(self, n_epochs, learning_rate, trainloader, testloader):
    start_time = time.time()
    logger.log.info(f"Training started at {start_time}...")

    criterion = nn.MSELoss(reduction="mean").to(self.device_name)
    optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
    t_losses, v_losses = [], []

    for epoch in range(n_epochs):
      epoch_training_start_time = time.time()

      train_loss, valid_loss = 0.0, 0.0
      self.model.train()

      for x, y in trainloader:
        optimizer.zero_grad()
        x = x.to(self.device_name)
        y = y.to(self.device_name)

        preds = self.model(x)

        loss = criterion(preds, y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

      epoch_loss = train_loss / len(trainloader)
      t_losses.append(epoch_loss)

      # validation step
      self.model.eval()
      for x, y in testloader:
        with torch.no_grad():
          x, y = x.to(self.device_name), y.squeeze().to(self.device_name)
          preds = self.model(x).squeeze()
          error = criterion(preds, y)
        valid_loss += error.item()
      valid_loss = valid_loss / len(testloader)
      v_losses.append(valid_loss)

      logger.log.info(f'{epoch} - train: {epoch_loss}, valid: {valid_loss}')
      epoch_training_time = time.time() - epoch_training_start_time
      logger.log.info(f"Epoch {epoch} training time: {epoch_training_time}")

    training_time = time.time() - start_time
    logger.log.info("Training time: {}".format(training_time))

    return t_losses, v_losses
