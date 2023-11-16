import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device_name):
        super(GRU, self).__init__()
        self.device_name = device_name
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        h0 = h0.to(self.devdevice_nameice)
        x = x.to(self.device_name)
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out
