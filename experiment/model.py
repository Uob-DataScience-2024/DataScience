import torch
from torch import nn
import torch.nn.functional as F


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_first=True, num_layers=6, dropout=0.25):
        super(Seq2Seq, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=batch_first, num_layers=num_layers, dropout=dropout)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=batch_first, num_layers=num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch, seq, feature]
        x, _ = self.encoder(x)
        x, _ = self.decoder(x)
        x = self.linear(x)
        return x
