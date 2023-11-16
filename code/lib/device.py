import torch.mps

from lib import logger


class Device(object):
  def __init__(self):
    super().__init__()
    self.name = "cpu"
    if torch.backends.mps.is_available():
      mps_device = torch.device("mps")
      x = torch.ones(1, device=mps_device)
      logger.log.info(x)
      # this ensures that the current current PyTorch installation was built with MPS activated.
      logger.log.info(torch.backends.mps.is_built())
      self.name = "mps"
      logger.log.info("Metal GPU is available.")
    elif torch.cuda.is_available():
      self.name = "cuda"
      logger.log.info("Nvidia GPU is available.")
    else:
      logger.log.info("GPU not found.")
