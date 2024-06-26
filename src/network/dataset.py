import math
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.utils

from data import GameData, GameNFLData, GamePlayData


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
    def __init__(self, data_dir, split=False, input_features: list = None, target_feature: str = None, max_seq_len=30000):
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
        self.input_feature_label = {}
        self.item_max_len = 0
        if self.input_features is not None:
            self.preprocess()
        self.label_map = self.build_label_map() if self.target_feature is not None else {}
        self.max_seq_len = max_seq_len

    def preprocess(self):
        self.input_feature_label = {col: [] for col in self.input_features}
        self.item_max_len = 0
        for game_id in tqdm(self.game_ids, desc='Preprocessing data', total=len(self.game_ids)):
            self.nfl_data[game_id].set_home_visitor(*self.home_visitor[game_id])
            tracking_data = self.nfl_data[game_id]
            for col in self.input_features:
                if pd.api.types.is_string_dtype(tracking_data.df[col]):
                    labels = tracking_data.df[col].unique().tolist()
                    self.input_feature_label[col] += labels
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
        if pd.api.types.is_numeric_dtype(type(temp[0])):
            if type(temp[0]) == int:
                temp = list(map(str, range(temp[0], temp[-1] + 1)))
            else:
                temp = list(map(str, temp))
        return temp

    def label_size(self):
        return len(self.label_map)

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
        category_labels_overwrite = {self.target_feature: self.label_map, **self.input_feature_label} if self.target_feature is not None else self.input_feature_label
        data, _ = tracking_data.tensor(resize_range_overwrite={}, category_labels_overwrite=category_labels_overwrite,
                                       columns=self.input_features + [self.target_feature] if self.input_features is not None and self.target_feature is not None else None, play_id_filter=partition)
        if self.input_features is not None and self.target_feature is not None:
            features = data[:, :len(self.input_features)]
            target = data[:, len(self.input_features):]
        else:
            features = data[:, :-1]
            target = data[:, -1]
        # one hot encoding for label
        target = torch.nn.functional.one_hot(target.to(torch.int64) + 1, num_classes=self.label_size() + 1).squeeze(1)
        return features[:self.max_seq_len], target[:self.max_seq_len]


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
        self.input_feature_label = {col: [] for col in self.input_features} if self.input_features is not None else {}
        self.item_max_len = 0
        for game_id in tqdm(self.game_ids, desc='Preprocessing data', total=len(self.game_ids)):
            self.nfl_data[game_id].set_home_visitor(*self.home_visitor[game_id])
            tracking_data = self.nfl_data[game_id]
            masks = tracking_data.union_id_mask()
            for union_id, mask in masks.items():
                self.cache.append((game_id, union_id, mask))
                self.item_max_len = max(self.item_max_len, len(mask))
            if self.input_features is not None:
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
        self.input_feature_label = {col: list(set(self.input_feature_label[col])) for col in self.input_features if len(self.input_feature_label[col]) > 0} if self.input_features is not None else {}

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
        target = torch.nn.functional.one_hot(target.to(torch.int64) + 1, num_classes=self.label_size() + 1).squeeze(0)
        return features, target


class NonTimeRelatedDataset(Dataset):
    def __init__(self, data_dir, split=False, input_features: list = None, target_feature: str = None, numpy_output=False):
        self.access_credit = None
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
        self.input_feature_label = {}
        self.item_max_len = 0
        if self.input_features is not None:
            self.preprocess()
        self.label_map = self.build_label_map() if self.target_feature is not None else {}
        self.numpy_output = numpy_output

    def preprocess(self):
        self.input_feature_label = {col: [] for col in self.input_features}
        self.item_max_len = 0
        self.access_credit = []
        for game_id in tqdm(self.game_ids, desc='Preprocessing data', total=len(self.game_ids)):
            self.nfl_data[game_id].set_home_visitor(*self.home_visitor[game_id])
            tracking_data = self.nfl_data[game_id]
            for col in self.input_features:
                if pd.api.types.is_string_dtype(tracking_data.df[col]):
                    labels = tracking_data.df[col].unique().tolist()
                    self.input_feature_label[col] += labels
            time_group = tracking_data.df.sort_values('time')

            # time_group['time'] = time_group['time'].astype(str)
            # time_group = time_group.groupby('time')
            # time_group = time_group.apply(lambda x: x.index.values)

            time_group['time_diff'] = time_group['time'].diff().dt.total_seconds()
            threshold = 10
            time_group['group'] = (time_group['time_diff'] > threshold).cumsum()
            group_indices = time_group.groupby('group').apply(lambda x: x.index.tolist())
            for group, index_arr in group_indices.items():
                index_arrs = time_group[time_group.index.isin(index_arr)].groupby('time').apply(lambda x: x.index.tolist())
                self.access_credit.append((game_id, group, index_arrs.values[0]))
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
        if pd.api.types.is_numeric_dtype(type(temp[0])):
            if type(temp[0]) == int:
                temp = list(map(str, range(temp[0], temp[-1] + 1)))
            else:
                temp = list(map(str, temp))
        return temp

    def label_size(self):
        return len(self.label_map)

    def __len__(self):
        return len(self.access_credit)

    def __getitem__(self, idx):
        game_id, group, mask = self.access_credit[idx]
        tracking_data = self.nfl_data[game_id]
        tracking_data.set_home_visitor(*self.home_visitor[game_id])
        # mask = tracking_data.df.index.isin(range(start, end + 1))
        category_labels_overwrite = {self.target_feature: self.label_map, **self.input_feature_label} if self.target_feature is not None else self.input_feature_label
        data, _ = tracking_data.tensor(resize_range_overwrite={}, category_labels_overwrite=category_labels_overwrite,
                                       columns=self.input_features + [self.target_feature] if self.input_features is not None and self.target_feature is not None else None, mask=mask,
                                       dataframe_out=self.numpy_output, no_repeat=True)
        # TODO: Big Change !!!!!!!!!!!!
        if self.numpy_output:
            data = data.values
        if self.input_features is not None and self.target_feature is not None:
            features = data[:, :len(self.input_features)]
            target = data[:, len(self.input_features):]
        else:
            features = data[:, :-1]
            target = data[:, -1]
            # one hot encoding for label
        if not self.numpy_output:
            target = torch.nn.functional.one_hot(target.to(torch.int64) + 1, num_classes=self.label_size() + 1).squeeze(1)
            return features, target
        else:
            return features, target
