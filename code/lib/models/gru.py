import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, config, device_name):
        super(GRU, self).__init__()
        self.device_name = device_name
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers

        self.gru = nn.GRU(config.input_dim, config.hidden_dim, config.num_layers, batch_first=True)
        self.fc = nn.Linear(config.idden_dim, config.output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        h0 = h0.to(self.device_name)
        x = x.to(self.device_name)
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out
