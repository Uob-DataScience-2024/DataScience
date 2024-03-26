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
