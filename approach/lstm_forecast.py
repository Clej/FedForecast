import torch
from torch import nn

# class for one hidden layer LSTM
class ShallowForecastLSTM(nn.Module):
    def __init__(self, num_var, hidden_units):
        super().__init__()
        self.num_var = num_var  # number of time series components
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=num_var,
            hidden_size=hidden_units,
            bias=True,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=num_var)

    def forward(self, x):
        batch_size = x.shape[0] # first dimension of a tensor returned by a DataLoader
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units)

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten(start_dim=0, end_dim=0)  # First dim of hn is num_layers, which is set to 1 above.

        return out