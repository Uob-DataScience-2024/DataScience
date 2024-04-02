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


class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_first=True, num_layers=6, dropout=0.25):
        super(Seq2SeqLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=batch_first, num_layers=num_layers, dropout=dropout)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=batch_first, num_layers=num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, encoder_hidden=None, decoder_hidden=None, first=False):
        # x: [batch, seq, feature]
        out_hidden = (encoder_hidden is not None and decoder_hidden is not None) or first
        x, _ = self.encoder(x, encoder_hidden)
        x, _ = self.decoder(x, decoder_hidden)
        x = self.linear(x)
        return x, (encoder_hidden, decoder_hidden) if out_hidden else x, None, None


class Seq2SeqGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_first=True, num_layers=6, dropout=0.25):
        super(Seq2SeqGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=batch_first, num_layers=num_layers, dropout=dropout)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=batch_first, num_layers=num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, encoder_hidden=None, decoder_hidden=None, first=False):
        # x: [batch, seq, feature]
        out_hidden = (encoder_hidden is not None and decoder_hidden is not None) or first
        x, _ = self.encoder(x, encoder_hidden)
        x, _ = self.decoder(x, decoder_hidden)
        x = self.linear(x)
        return x, (encoder_hidden, decoder_hidden) if out_hidden else x, None, None


class SameSizeCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, input_shape=(203, 14), num_layers=5, dropout=0.25):
        super(SameSizeCNN, self).__init__()
        self.hidden_dim = hidden_dim
        convs = []
        convs.append(nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1))
        for i in range(num_layers):
            convs.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1))
            convs.append(nn.ReLU())
        self.feature = nn.Sequential(*convs)
        self.avg_pool = nn.AdaptiveAvgPool2d(8)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
