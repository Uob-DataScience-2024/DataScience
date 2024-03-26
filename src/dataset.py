import math
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.utils

from game_data import GameData
from playdata import GamePlayData
from trackingdata import GameTrackingData
from datastructure import GameNFLData


class TrackingDataset(Dataset):
    def __init__(self, data_dir):
        weeks = [x for x in os.listdir(data_dir) if x.startswith('week')]
        if len(weeks) == 0:
            raise ValueError("No week file found")
        weeks = [os.path.join(data_dir, x) for x in weeks]
        play_file = os.path.join(data_dir, 'plays.csv')
        games_file = os.path.join(data_dir, 'games.csv')
        pff_file = os.path.join(data_dir, 'pffScoutingData.csv')
        self.play_data = GamePlayData.load(play_file)
        self.nfl_data = GameNFLData.loads(weeks, pff_file, play_file)
        self.games = list(self.nfl_data.keys())
        self.spilt_data = {
            key: value.get_quarter_partition() for key, value in self.play_data.items()
        }
        self.win = {
            key: value.win_home() for key, value in self.play_data.items()
        }
        self.games_data = GameData(games_file)
        self.home_visitor = self.games_data.get_home_visitor()
        for gameId in self.nfl_data:
            self.nfl_data[gameId].set_home_visitor(*self.home_visitor[gameId])

    # def __len__(self):
    #     return len(self.play_data) * 4
    #
    # def __getitem__(self, idx):
    #     quarter = int(idx % 4)
    #     idx = int(math.floor(idx / 4.0))
    #     data = self.tracking_data[self.games[idx]]
    #     partition = self.spilt_data[self.games[idx]][quarter]
    #     data = data.tensor(resize_range_overwrite={}, category_labels_overwrite={}, play_id_filter=partition)
    #     return data

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        quarter = int(idx % 4)
        idx = int(math.floor(idx / 4.0))
        data = self.nfl_data[self.games[idx]]
        label = int(self.win[self.games[idx]])
        data, _ = data.tensor(resize_range_overwrite={}, category_labels_overwrite={})
        # one hot encoding for label
        label = torch.tensor([label], dtype=torch.float32)
        label = torch.nn.functional.one_hot(label.to(torch.int64), num_classes=2).squeeze(0)
        return data, label.to(torch.float32)


class SequenceDataset(Dataset):
    def __init__(self, data_dir, split=False, input_features: list = None, target_feature: str = None):
        weeks = [x for x in os.listdir(data_dir) if x.startswith('week')]
        if len(weeks) == 0:
            raise ValueError("No week file found")
        weeks = [os.path.join(data_dir, x) for x in weeks]
        play_file = os.path.join(data_dir, 'plays.csv')
        pff_file = os.path.join(data_dir, 'pffScoutingData.csv')
        games_file = os.path.join(data_dir, 'games.csv')
        self.nfl_data = GameNFLData.loads(weeks, pff_file, play_file)
        self.game_data = GameData(games_file)
        self.game_ids = list(self.nfl_data.keys())
        self.home_visitor = self.game_data.get_home_visitor()
        self.split = split
        self.spilt_data = {
            key: value.get_quarter_partition() for key, value in self.nfl_data.items()
        }
        self.input_features = input_features
        self.target_feature = target_feature

    def __len__(self):
        return len(self.game_ids) * 4 if self.split else len(self.game_ids)

    def __getitem__(self, idx):
        if self.split:
            quarter = int(idx % 4)
            idx = int(math.floor(idx / 4.0))
        else:
            quarter = None
        game_id = self.game_ids[idx]
        tracking_data = self.nfl_data[game_id]
        tracking_data.set_home_visitor(*self.home_visitor[game_id])
        partition = self.spilt_data[game_id][quarter] if self.split else None
        data, _ = tracking_data.tensor(resize_range_overwrite={}, category_labels_overwrite={},
                                       columns=self.input_features + [self.target_feature] if self.input_features is not None and self.target_feature is not None else None, play_id_filter=partition)
        if self.input_features is not None and self.target_feature is not None:
            features = data[:, :len(self.input_features)]
            target = data[:, len(self.input_features):]
        else:
            features = data[:, :-1]
            target = data[:, -1]
        return features, target


class SegmentDataset(Dataset):
    def __init__(self, data_dir, input_features: list = None, target_feature: str = None):
        weeks = [x for x in os.listdir(data_dir) if x.startswith('week')]
        if len(weeks) == 0:
            raise ValueError("No week file found")
        weeks = [os.path.join(data_dir, x) for x in weeks]
        play_file = os.path.join(data_dir, 'plays.csv')
        pff_file = os.path.join(data_dir, 'pffScoutingData.csv')
        games_file = os.path.join(data_dir, 'games.csv')
        self.nfl_data = GameNFLData.loads(weeks, pff_file, play_file)
        self.game_data = GameData(games_file)
        self.game_ids = list(self.nfl_data.keys())
        self.home_visitor = self.game_data.get_home_visitor()
        self.input_features = input_features
        self.target_feature = target_feature
        self.data = {}
        self.cache = []
        self.input_feature_label = None
        self.item_max_len = 0
        self.preprocess()
        self.label_map = self.build_label_map()

    def preprocess(self):
        self.input_feature_label = {col: [] for col in self.input_features}
        self.item_max_len = 0
        for game_id in tqdm(self.game_ids, desc='Preprocessing data', total=len(self.game_ids)):
            self.nfl_data[game_id].set_home_visitor(*self.home_visitor[game_id])
            tracking_data = self.nfl_data[game_id]
            masks = tracking_data.union_id_mask()
            for union_id, mask in masks.items():
                self.cache.append((game_id, union_id, mask))
                self.item_max_len = max(self.item_max_len, len(mask))
            for col in self.input_features:
                if pd.api.types.is_string_dtype(tracking_data.df[col]):
                    labels = tracking_data.df[col].unique().tolist()
                    self.input_feature_label[col] += labels
            # self.data[game_id] = []
            # for union_id, mask in masks.items():
            #     data, _ = tracking_data.tensor(resize_range_overwrite={}, category_labels_overwrite={},
            #                                    columns=self.input_features + [self.target_feature] if self.input_features is not None and self.target_feature is not None else None,
            #                                    mask=mask)
            #     if self.input_features is not None and self.target_feature is not None:
            #         features = data[:, :len(self.input_features)]
            #         target = data[:, len(self.input_features):]
            #     else:
            #         features = data[:, :-1]
            #         target = data[:, -1]
            #     self.cache.append((features, target))
            #     self.data[game_id].append(len(self.cache) - 1)
        self.input_feature_label = {col: list(set(self.input_feature_label[col])) for col in self.input_features if len(self.input_feature_label[col]) > 0}

    def build_label_map(self):
        temp = []
        for game_id in self.game_ids:
            df = self.nfl_data[game_id].df
            col = self.target_feature if self.target_feature is not None else df.columns[-1]
            label = df[col].unique()
            for i in label:
                if i not in temp and not pd.isna(i):
                    temp.append(i)
        temp.sort()
        return temp

    def label_size(self):
        return len(self.label_map)

    def __len__(self):
        return len(self.cache)

    def padding(self, data: torch.Tensor):
        pad_rows = (self.item_max_len - data.size(0)) // 2
        padded_data = torch.nn.functional.pad(data, (0, 0, pad_rows, pad_rows), value=0)
        return padded_data

    def __getitem__(self, idx):
        game_id, union_id, mask = self.cache[idx]
        tracking_data = self.nfl_data[game_id]
        category_labels_overwrite = {self.target_feature: self.label_map, **self.input_feature_label} if self.target_feature is not None else self.input_feature_label
        data, _ = tracking_data.tensor(resize_range_overwrite={}, category_labels_overwrite=category_labels_overwrite,
                                       columns=self.input_features + [self.target_feature] if self.input_features is not None and self.target_feature is not None else None,
                                       mask=mask)
        if self.input_features is not None and self.target_feature is not None:
            features = data[:, :len(self.input_features)]
            target = data[:, len(self.input_features):][0]
        else:
            features = data[:, :-1]
            target = data[:, -1][0]
        # one hot encoding for label
        if features.size(0) < self.item_max_len:
            features = self.padding(features)
        target = torch.nn.functional.one_hot(target.to(torch.int64) + 1, num_classes=self.label_size()).squeeze(0)
        return features, target
