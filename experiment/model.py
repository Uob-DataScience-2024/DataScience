import math

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Transformer


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


class Seq2SeqGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_first=True, num_layers=6, dropout=0.25):
        super(Seq2SeqGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=batch_first, num_layers=num_layers, dropout=dropout)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=batch_first, num_layers=num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch, seq, feature]
        x, _ = self.encoder(x)
        x, _ = self.decoder(x)
        x = self.linear(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 input_feature_size: int = 12,
                 output_feature_size: int = 13,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, output_feature_size)
        self.post_mapping = nn.Linear(input_feature_size, emb_size)
        self.post_mapping_tgt = nn.Linear(output_feature_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, maxlen=20000)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor=None,
                tgt_mask: Tensor=None,
                src_padding_mask: Tensor=None,
                tgt_padding_mask: Tensor=None,
                memory_key_padding_mask: Tensor=None):
        src_emb = self.positional_encoding(self.post_mapping(src))
        tgt_emb = self.positional_encoding(self.post_mapping_tgt(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.post_mapping(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.post_mapping_tgt(tgt)), memory, tgt_mask)
