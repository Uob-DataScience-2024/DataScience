import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2 * num_layers, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        out, hidden = self.lstm(x)
        x = torch.concat([hidden[0], hidden[1]], dim=1)
        x = x.view(out.size(0), -1)
        out = self.fc(x)
        return out
