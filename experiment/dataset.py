import json
import math
import os.path
from datetime import datetime

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import transforms
import torch.nn.functional as F
from tqdm.rich import tqdm


class DatasetPffBlockType(Dataset):
    def __init__(self, data_dir, pff_filename='pffScoutingData.csv', tracking_filename='week{}.csv', cache=False,
                 x_col=['nflId_x', 'frameId', 'jerseyNumber', 'team', 'playDirection', 'x', 'y', 's', 'a', 'dis', 'o', 'dir'],
                 x_category_col=['nflId_x', 'jerseyNumber', 'team', 'playDirection'],
                 x_self_category_col=['nflId_x', 'team'],
                 y_col='pff_blockType'):
        self.category_table = None
        self.data_dir = data_dir
        self.pff_filename = pff_filename
        self.tracking_filename = tracking_filename
        self.final_data = {}
        self.raw_data = {
            'pff': self.load_pff_data(),
            'tracking': self.load_tracking_data()
        }
        self.postprocess(cache)
        self.total_data = [self.final_data[game] for game in self.final_data]
        self.x_col = x_col
        self.x_category_col = x_category_col
        self.x_self_category_col = x_self_category_col
        self.y_col = y_col
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.build_category_table()

    def load_pff_data(self):
        return pd.read_csv(os.path.join(self.data_dir, self.pff_filename))

    def load_tracking_data(self, max_week=8):
        datas = []
        for i in range(1, max_week + 1):
            file = os.path.join(self.data_dir, self.tracking_filename.format(i))
            if os.path.exists(file) and os.path.isfile(file):
                datas.append(pd.read_csv(os.path.join(self.data_dir, self.tracking_filename.format(i))))
        # concat all the dataframes
        return pd.concat(datas, ignore_index=True)

    def load_cache(self):
        if not os.path.exists(os.path.join(self.data_dir, 'cache') or not os.path.isdir(os.path.join(self.data_dir, 'cache'))):
            os.mkdir(os.path.join(self.data_dir, 'cache'))
        try:
            with open(os.path.join(self.data_dir, 'cache', 'cache_list.json'), 'r') as f:
                cache_list = json.load(f)
                self.final_data = {game.split('.')[0]: {} for game in cache_list}
        except FileNotFoundError:
            return False
        csvs = [f for f in os.listdir(os.path.join(self.data_dir, 'cache')) if f.endswith('.csv')]
        if len(csvs) != len(cache_list):
            return False
        for game in self.final_data.keys():
            cache_file = os.path.join(self.data_dir, 'cache', f'{game}.csv')
            if os.path.exists(cache_file) and os.path.isfile(cache_file):
                self.final_data[game] = pd.read_csv(cache_file)
            else:
                return False
        return True

    def save_cache(self):
        if not os.path.exists(os.path.join(self.data_dir, 'cache') or not os.path.isdir(os.path.join(self.data_dir, 'cache'))):
            os.mkdir(os.path.join(self.data_dir, 'cache'))
        names = [f'{game}.csv' for game in self.final_data]
        with open(os.path.join(self.data_dir, 'cache', 'cache_list.json'), 'w') as f:
            json.dump(names, f, indent=4)
        for game, data in self.final_data.items():
            data.to_csv(os.path.join(self.data_dir, 'cache', f'{game}.csv'), index=False)

    def postprocess(self, cache=False):
        if cache:
            logger.info('use cache')
            if self.load_cache():
                return
            logger.info('cache not found, start to process data')
        data_in_each_game = {game: {} for game in self.raw_data['tracking']['gameId'].unique()}
        for game in data_in_each_game:
            data_in_each_game[game] = {
                'pff': self.raw_data['pff'][self.raw_data['pff']['gameId'] == game],
                'tracking': self.raw_data['tracking'][self.raw_data['tracking']['gameId'] == game]
            }
        self.final_data = {game: {} for game in data_in_each_game}
        for game, data in data_in_each_game.items():
            pff = data['pff'].copy()
            tracking = data['tracking'].copy()
            # create union id by combining nflId and playId for tracking data and pff data
            # remove na for nflId and playId
            tracking = tracking.dropna(subset=['nflId', 'playId'])
            pff = pff.dropna(subset=['nflId', 'playId'])
            tracking['union_id'] = tracking['nflId'].astype(int).astype(str) + '_' + tracking['playId'].astype(int).astype(str)
            pff['union_id'] = pff['nflId'].astype(int).astype(str) + '_' + pff['playId'].astype(int).astype(str)
            # create a new dataframe that concat tracking and pff data columns
            # and copy the pff data to each line which have same union_id, because the pff data is less than tracking data
            # so we need to copy the pff data to each line which have same union_id
            result = tracking.merge(pff, on='union_id', how='left')
            self.final_data[game] = result
        if cache:
            self.save_cache()

    def statistics(self):
        # statistics target:
        # 1. pff available rate
        # 2. pff_blockType available rate
        # 3. number of games
        # 4. number of data

        number_of_games = len(self.final_data)
        number_of_data = sum([len(self.final_data[game]) for game in self.final_data])
        pff_available = 0
        pff_blockType_available = 0
        total_lines = 0
        for game, data in self.final_data.items():
            total_lines += len(data)
            pff_available += data.dropna(subset=['pff_role']).shape[0]
            pff_blockType_available += data.dropna(subset=['pff_blockType']).shape[0]
        return {
            # 'raw_pff_available_rate': raw_pff_available_rate,
            # 'raw_pff_blockType_available_rate': raw_pff_blockType_available_rate,
            'number_of_games': number_of_games,
            'number_of_data': number_of_data,
            'pff_available_rate': pff_available / total_lines,
            'pff_blockType_available_rate': pff_blockType_available / total_lines
        }

    def get_column(self):
        # return the column dict, key: name, value: type
        return {col: self.final_data[list(self.final_data.keys())[0]][col].dtype for col in self.final_data[list(self.final_data.keys())[0]].columns}

    def get_categories(self, category):
        d = []
        for game, data in self.final_data.items():
            d.extend(data[category].unique())
        return list(set(d))

    def build_category_table(self):
        # build category table for each category column
        category_table = {}
        for col in self.x_col + [self.y_col]:
            if col in self.x_category_col or col == self.y_col:
                category_table[col] = self.get_categories(col)
                # if nan in the category, move to the first
                if np.nan in category_table[col]:
                    category_table[col].remove(np.nan)
                    category_table[col].sort()
                    category_table[col].insert(0, np.nan)
                else:
                    category_table[col].sort()
        self.category_table = category_table
        return category_table

    def __len__(self):
        return len(self.total_data)

    def __getitem__(self, idx):
        data = self.total_data[idx]
        category_table = {}
        results = []
        for name in self.x_col:
            if name in self.x_category_col:
                if name in self.x_self_category_col:
                    results.append(data[name].astype('category').cat.codes)
                else:
                    results.append(pd.Categorical(data[name], categories=self.category_table[name]).codes)
            else:
                results.append(data[name])
        raw_label = data[self.y_col].to_numpy().tolist()
        label = [self.category_table[self.y_col].index(i) if type(i) == str else 0 for i in raw_label]
        results = torch.tensor(np.array(results).T, dtype=torch.float32)
        label = torch.tensor(np.array(label))
        # results = self.transform(results)
        return results, F.one_hot(label.long(), num_classes=len(self.category_table[self.y_col])).to(torch.float32)


class DatasetPffBlockTypeAutoSpilt(Dataset):
    def __init__(self, data_dir, pff_filename='pffScoutingData.csv', tracking_filename='week{}.csv', play_data_filename="plays.csv", cache=False,
                 x_col=['nflId_x', 'frameId', 'jerseyNumber', 'team', 'playDirection', 'x', 'y', 's', 'a', 'dis', 'o', 'dir'],
                 x_category_col=['nflId_x', 'jerseyNumber', 'team', 'playDirection'],
                 x_self_category_col=['nflId_x', 'team'],
                 y_col='pff_blockType'):
        self.category_table = None
        self.data_dir = data_dir
        self.pff_filename = pff_filename
        self.tracking_filename = tracking_filename
        self.play_data_filename = play_data_filename
        self.final_data = {}
        self.raw_data = {
            'pff': self.load_pff_data(),
            'tracking': self.load_tracking_data(),
            'play': self.load_play_data()
        }
        self.postprocess(cache)
        self.total_data = [self.final_data[game] for game in self.final_data]
        self.x_col = x_col
        self.x_category_col = x_category_col
        self.x_self_category_col = x_self_category_col
        self.y_col = y_col
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.build_category_table()

    def load_pff_data(self):
        return pd.read_csv(os.path.join(self.data_dir, self.pff_filename))

    def load_tracking_data(self, max_week=8):
        datas = []
        for i in range(1, max_week + 1):
            file = os.path.join(self.data_dir, self.tracking_filename.format(i))
            if os.path.exists(file) and os.path.isfile(file):
                datas.append(pd.read_csv(os.path.join(self.data_dir, self.tracking_filename.format(i))))
        # concat all the dataframes
        return pd.concat(datas, ignore_index=True)

    def load_play_data(self):
        return pd.read_csv(os.path.join(self.data_dir, self.play_data_filename))

    def load_cache(self):
        if not os.path.exists(os.path.join(self.data_dir, 'cache') or not os.path.isdir(os.path.join(self.data_dir, 'cache'))):
            os.mkdir(os.path.join(self.data_dir, 'cache'))
        try:
            with open(os.path.join(self.data_dir, 'cache', 'cache_list.json'), 'r') as f:
                cache_list = json.load(f)
                self.final_data = {game.split('.')[0]: {} for game in cache_list}
        except FileNotFoundError:
            return False
        csvs = [f for f in os.listdir(os.path.join(self.data_dir, 'cache')) if f.endswith('.csv')]
        if len(csvs) != len(cache_list):
            return False
        for game in tqdm(self.final_data.keys(), desc='load cache'):
            cache_file = os.path.join(self.data_dir, 'cache', f'{game}.csv')
            if os.path.exists(cache_file) and os.path.isfile(cache_file):
                self.final_data[game] = pd.read_csv(cache_file, low_memory=False)
            else:
                return False
        return True

    def save_cache(self):
        if not os.path.exists(os.path.join(self.data_dir, 'cache') or not os.path.isdir(os.path.join(self.data_dir, 'cache'))):
            os.mkdir(os.path.join(self.data_dir, 'cache'))
        names = [f'{game}.csv' for game in self.final_data]
        with open(os.path.join(self.data_dir, 'cache', 'cache_list.json'), 'w') as f:
            json.dump(names, f, indent=4)
        for game, data in self.final_data.items():
            data.to_csv(os.path.join(self.data_dir, 'cache', f'{game}.csv'), index=False)

    def postprocess(self, cache=False):
        if cache:
            logger.info('use cache')
            if self.load_cache():
                return
            logger.info('cache not found, start to process data')
        data_in_each_game = {game: {} for game in self.raw_data['tracking']['gameId'].unique()}
        for game in tqdm(data_in_each_game, desc='split table by each game'):
            data_in_each_game[game] = {
                'pff': self.raw_data['pff'][self.raw_data['pff']['gameId'] == game],
                'tracking': self.raw_data['tracking'][self.raw_data['tracking']['gameId'] == game],
                'play': self.raw_data['play'][self.raw_data['play']['gameId'] == game]
            }
        self.final_data = {game: {} for game in data_in_each_game}
        for game, data in tqdm(data_in_each_game.items(), desc='merge table for each game'):
            pff = data['pff'].copy()
            tracking = data['tracking'].copy()
            # create union id by combining nflId and playId for tracking data and pff data
            # remove na for nflId and playId
            tracking = tracking.dropna(subset=['nflId', 'playId'])
            pff = pff.dropna(subset=['nflId', 'playId'])
            play = data['play'].dropna(subset=['playId'])
            tracking['union_id'] = tracking['nflId'].astype(int).astype(str) + '_' + tracking['playId'].astype(int).astype(str)
            pff['union_id'] = pff['nflId'].astype(int).astype(str) + '_' + pff['playId'].astype(int).astype(str)
            # create a new dataframe that concat tracking and pff data columns
            # and copy the pff data to each line which have same union_id, because the pff data is less than tracking data
            # so we need to copy the pff data to each line which have same union_id
            result = tracking.merge(play, on='playId', how='left')
            result = result.merge(pff, on='union_id', how='left')
            self.final_data[game] = result
        if cache:
            self.save_cache()

    def statistics(self):
        # statistics target:
        # 1. pff available rate
        # 2. pff_blockType available rate
        # 3. number of games
        # 4. number of data

        number_of_games = len(self.final_data)
        number_of_data = sum([len(self.final_data[game]) for game in self.final_data])
        pff_available = 0
        pff_blockType_available = 0
        total_lines = 0
        for game, data in self.final_data.items():
            total_lines += len(data)
            pff_available += data.dropna(subset=['pff_role']).shape[0]
            pff_blockType_available += data.dropna(subset=['pff_blockType']).shape[0]
        return {
            # 'raw_pff_available_rate': raw_pff_available_rate,
            # 'raw_pff_blockType_available_rate': raw_pff_blockType_available_rate,
            'number_of_games': number_of_games,
            'number_of_data': number_of_data,
            'pff_available_rate': pff_available / total_lines,
            'pff_blockType_available_rate': pff_blockType_available / total_lines
        }

    def get_column(self):
        # return the column dict, key: name, value: type
        return {col: self.final_data[list(self.final_data.keys())[0]][col].dtype for col in self.final_data[list(self.final_data.keys())[0]].columns}

    def get_categories(self, category):
        d = []
        for game, data in self.final_data.items():
            d.extend(data[category].unique())
        return list(set(d))

    def build_category_table(self):
        # build category table for each category column
        category_table = {}
        for col in self.x_col + [self.y_col]:
            if col in self.x_category_col or col == self.y_col:
                category_table[col] = self.get_categories(col)
                # if nan in the category, move to the first
                if np.nan in category_table[col]:
                    category_table[col].remove(np.nan)
                    category_table[col].sort()
                    category_table[col].insert(0, np.nan)
                else:
                    category_table[col].sort()
        self.category_table = category_table
        return category_table

    def __len__(self):
        return len(self.total_data) * 4

    def __getitem__(self, idx):
        idx = idx / 4.0
        data = self.total_data[int(math.floor(idx))]
        quarter = int(idx % 4) + 1
        data = data[data['quarter'] == quarter]
        category_table = {}
        results = []
        for name in self.x_col:
            if name in self.x_category_col:
                if name in self.x_self_category_col:
                    results.append(data[name].astype('category').cat.codes)
                else:
                    results.append(pd.Categorical(data[name], categories=self.category_table[name]).codes)
            else:
                results.append(data[name])
        raw_label = data[self.y_col].to_numpy().tolist()
        label = [self.category_table[self.y_col].index(i) if type(i) == str else 0 for i in raw_label]
        results = torch.tensor(np.array(results).T, dtype=torch.float32)
        label = torch.tensor(np.array(label))
        # results = self.transform(results)
        return results, F.one_hot(label.long(), num_classes=len(self.category_table[self.y_col])).to(torch.float32)
