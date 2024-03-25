import math
import os

import torch
from torch.utils.data import Dataset

from playdata import GamePlayData
from trackingdata import GameTrackingData


class TrackingDataset(Dataset):
    def __init__(self, data_dir):
        weeks = [x for x in os.listdir(data_dir) if x.startswith('week')]
        if len(weeks) == 0:
            raise ValueError("No week file found")
        weeks = [os.path.join(data_dir, x) for x in weeks]
        play_file = os.path.join(data_dir, 'plays.csv')
        self.play_data = GamePlayData.load(play_file)
        self.tracking_data = GameTrackingData.load_all(weeks)
        self.games = list(self.tracking_data.keys())
        self.spilt_data = {
            key: value.get_quarter_partition() for key, value in self.play_data.items()
        }
        self.win = {
            key: value.win_home() for key, value in self.play_data.items()
        }

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
        return len(self.play_data)

    def __getitem__(self, idx):
        quarter = int(idx % 4)
        idx = int(math.floor(idx / 4.0))
        data = self.tracking_data[self.games[idx]]
        label = int(self.win[self.games[idx]])
        data, _ = data.tensor(resize_range_overwrite={}, category_labels_overwrite={})
        # one hot encoding for label
        label = torch.tensor([label], dtype=torch.float32)
        label = torch.nn.functional.one_hot(label.to(torch.int64), num_classes=2).squeeze(0)
        return data, label
