import math
import os

import torch
from torch.utils.data import Dataset

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
        data, _ = tracking_data.tensor(resize_range_overwrite={}, category_labels_overwrite={}, columns=self.input_features + [self.target_feature] if self.input_features is not None and self.target_feature is not None else None, play_id_filter=partition)
        if self.input_features is not None and self.target_feature is not None:
            features = data[:, :len(self.input_features)]
            target = data[:, len(self.input_features):]
        else:
            features = data[:, :-1]
            target = data[:, -1]
        return features, target
