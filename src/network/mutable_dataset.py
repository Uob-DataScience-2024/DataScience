import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.total_data import SplitMode, TrackingNormData, PffNormData, PlayNormData, PlayerNormData, GameNormData, MergeNormData


class SimpleDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.long)


class DataGenerator:
    def __init__(self, tracking: TrackingNormData, pff: PffNormData, play: PlayNormData, game: GameNormData, player: PlayerNormData, merge: MergeNormData):
        self.tracking = tracking
        self.pff = pff
        self.play = play
        self.game = game
        self.player = player
        self.merge = merge

    def generate_dataset(self, x_columns, y_column, data_type='numpy', data_type_mapping=None, norm=False, player_needed=False, game_needed=False):
        if data_type_mapping is None:
            data_type_mapping = {}
        df = self.merge.game.copy()
        if player_needed:
            df = df.merge(self.merge.player, on='nflId', how='left')
            df = df.dropna(subset=['height'])
        if game_needed:
            df = df.merge(self.merge.game_info, on='gameId', how='left')

        preprocess_cols = x_columns + [y_column]
        cols_type = {name: df.dtypes[name] for name in df.columns}

        df = df.fillna(method='ffill')
        for col, dtype in {c: cols_type[c] for c in x_columns}.items():
            if col in data_type_mapping:
                df[col] = df[col].apply(data_type_mapping[col])
                continue
            if pd.api.types.is_string_dtype(dtype):
                df[col] = df[col].astype('category').cat.codes
            if pd.api.types.is_numeric_dtype(dtype):
                if norm:
                    df[col] = (df[col] - np.min(df[col])) / (np.max(df[col]) - np.min(df[col]))

        df = df.fillna(method='ffill')
        X = df[x_columns]
        Y = df[y_column]
        Y.dropna()
        if data_type == 'numpy':
            return X.to_numpy(), Y.to_numpy()
        elif data_type == 'pandas':
            return X, Y
        elif data_type == 'torch':
            return SimpleDataset(X.values, Y.astype('category').cat.codes.to_numpy())
        else:
            raise ValueError(f"Data type {data_type} not supported")