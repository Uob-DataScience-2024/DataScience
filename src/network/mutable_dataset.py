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


def heightInches(h):
    temp = list(map(int, h.split('-')))
    return temp[0] * 12 + temp[1]


def convertTimeToNumerical(t):
    t = t.split(":")
    if len(t) == 2:
        return int(t[0]) * 60 + int(t[1])
    return int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])


class DataGenerator:
    def __init__(self, tracking: TrackingNormData, pff: PffNormData, play: PlayNormData, game: GameNormData, player: PlayerNormData, merge: MergeNormData):
        self.tracking = tracking
        self.pff = pff
        self.play = play
        self.game = game
        self.player = player
        self.merge = merge

    def generate_dataset(self, x_columns, y_column, data_type='numpy', data_type_mapping=None, data_type_mapping_inverse=None, norm=False, tracking_data_include=True, player_needed=False,
                         game_needed=False, dropna_y=True,
                         with_mapping_log=False):
        if data_type_mapping is None:
            data_type_mapping = {'gameClock': convertTimeToNumerical, 'height': heightInches}
        if data_type_mapping_inverse is None:
            data_type_mapping_inverse = {'height': lambda x: f"{int(x // 12)}-{int(x % 12)}", 'gameClock': lambda x: f"{int(x // 60)}:{int(x % 60)}"}
        if tracking_data_include:
            df = self.merge.game.copy()
        else:
            df = self.play.data.copy().merge(self.pff.data, on=['gameId', 'playId'], how='left')
        if player_needed:
            df = df.merge(self.merge.player, on='nflId', how='left')
            df = df.dropna(subset=['height'])
        if game_needed:
            df = df.merge(self.merge.game_info, on='gameId', how='left')

        preprocess_cols = x_columns + [y_column]
        cols_type = {name: df.dtypes[name] for name in df.columns}
        mapping_log = {}

        df = df.fillna(method='ffill')
        for col, dtype in {c: cols_type[c] for c in preprocess_cols}.items():
            if col in data_type_mapping:
                df[col] = df[col].apply(data_type_mapping[col])
                mapping_log[col] = {'type': 'function', 'mapping': data_type_mapping[col]}
                continue
            if pd.api.types.is_string_dtype(dtype):
                labels = df[col].unique()
                labels.sort()
                cat_type = pd.CategoricalDtype(categories=labels, ordered=True)
                df[col] = df[col].astype(cat_type).cat.codes
                mapping_log[col] = {'type': 'category', 'mapping': {i: label for i, label in enumerate(labels)}}
            if pd.api.types.is_numeric_dtype(dtype):
                if norm and col != y_column:
                    mapping_log[col] = {'type': 'numeric', 'mapping': {'min': np.min(df[col]), 'max': np.max(df[col])}}
                    df[col] = (df[col] - np.min(df[col])) / (np.max(df[col]) - np.min(df[col]))

        df = df.fillna(method='ffill')
        if dropna_y:
            df = df.dropna(subset=[y_column])
        X = df[x_columns]
        Y = df[y_column]
        Y.dropna()
        if data_type == 'numpy':
            return (X.to_numpy(), Y.to_numpy()) if not with_mapping_log else (X.to_numpy(), Y.to_numpy(), mapping_log, data_type_mapping_inverse)
        elif data_type == 'pandas':
            return (X, Y) if not with_mapping_log else (X, Y, mapping_log, data_type_mapping_inverse)
        elif data_type == 'torch':
            return SimpleDataset(X.values, Y.astype('category').cat.codes.to_numpy()) if not with_mapping_log else (
                SimpleDataset(X.values, Y.astype('category').cat.codes.to_numpy()), mapping_log, data_type_mapping_inverse)
        else:
            raise ValueError(f"Data type {data_type} not supported")
