import re
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger


class TrackingDataItem:
    x: float
    """x coordinate of the player (0 - 120)"""
    y: float
    """y coordinate of the player (0 - 53.3)"""
    s: float
    """speed of the player"""
    a: float
    """acceleration of the player"""
    dis: float
    """Distance traveled from prior time point, in yards"""
    o: float
    """Player orientation (0-360) (degrees)"""
    dir: float
    """Angle of player motion (0-360) (degrees)"""
    team: Optional[str]
    """Team code"""

    event: str
    """Event that occurred(not each line has an event)"""
    jerseyNumber: int

    no_payload_columns = {
        'game_id': 'gameId',
        'play_id': 'playId',
        'nfl_id': 'nflId',
        'frame_id': 'frameId',
        'dt': 'time',
    }

    resize_range = {
        'x': (0, 120),
        'y': (0, 53.3),
        's': None,
        'a': None,
        'dis': None,
        'o': (0, 360),
        'dir': (0, 360),
    }

    category_labels = {
        'event': None,
    }

    block_columns = ['gameId']

    def __init__(self, week: str, game_id: int, play_id: int, nfl_id: int, frame_id: int, dt: datetime, number_payload: dict, text_payload: dict):
        self.week = week
        self.game_id = game_id
        self.play_id = play_id
        self.nfl_id = nfl_id
        self.frame_id = frame_id
        self.time = dt
        self.number_payload = number_payload
        self.text_payload = text_payload
        for key, value in number_payload.items():
            setattr(self, key, value)
        for key, value in text_payload.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        return f"TrackingDataItem(GameID: {self.game_id}, PlayID: {self.play_id}, FrameID: {self.frame_id}, Time: {self.time}, NFLID: {self.nfl_id}, Event: {self.event}, x: {self.x}, y: {self.y}, s: {self.s}, a: {self.a}, dis: {self.dis}, o: {self.o}, dir: {self.dir})"


class GameTrackingData:
    def __init__(self, game_id: int, date_start: datetime, date_end: datetime, week: str, df: pd.DataFrame):
        self.home_visitor = None
        self.game_id = game_id
        self.date_start = date_start
        self.date_end = date_end
        self.week = week
        self.df = df
        columns = df.columns
        headers = {}  # save the type of each column
        for column in columns:
            headers[column] = df[column].dtype
        self.number_list = list(filter((lambda x: pd.api.types.is_numeric_dtype(x[1]) and x[0] not in TrackingDataItem.no_payload_columns.values()), headers.items()))
        self.text_list = list(filter((lambda x: not pd.api.types.is_numeric_dtype(x[1]) and x[0] not in TrackingDataItem.no_payload_columns.values()), headers.items()))
        self.no_payload_columns = list(TrackingDataItem.no_payload_columns.items())

    @staticmethod
    def load(filename):
        week = re.search(r'week(\d+)', filename).group(1)
        week = str(int(week))
        df = pd.read_csv(filename)
        df['time'] = pd.to_datetime(df['time'])
        loaded = {}
        for game_id in df['gameId'].unique():
            sub_df = df[df['gameId'] == game_id]
            date_start = sub_df['time'].min()
            date_end = sub_df['time'].max()
            loaded[game_id] = GameTrackingData(game_id, date_start, date_end, week, sub_df)
        return loaded

    @staticmethod
    def load_all(filenames):
        loaded = {}
        for filename in filenames:
            loaded.update(GameTrackingData.load(filename))
        return loaded

    def __str__(self) -> str:
        return f"GameTrackingData(GameID:{self.game_id}, Week:{self.week}, DTR:[{self.date_start} -> {self.date_end}])[{len(self.df)}]"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> TrackingDataItem:
        line = self.df.iloc[idx]
        args = {arg_name: line[col_name] for arg_name, col_name in self.no_payload_columns}
        args['number_payload'] = {col_name: line[col_name] for col_name, dtype in self.number_list}
        args['text_payload'] = {col_name: line[col_name] for col_name, dtype in self.text_list}
        return TrackingDataItem(self.week, **args)

    def set_home_visitor(self, home, visitor):
        self.home_visitor = [home, visitor]

    def statistics(self):
        return self.df.describe()

    def tensor(self, resize_range_overwrite: dict, category_labels_overwrite: dict, columns: list[str] = None, dtype=torch.float32, play_id_filter=None) -> [torch.Tensor, dict[str, dict]]:
        df = self.df.copy()
        if play_id_filter is not None:
            df = df[df['playId'].isin(play_id_filter)]
        if self.home_visitor is not None:
            category_labels_overwrite['team'] = self.home_visitor
        if columns is None:
            columns = df.columns
        elif len(list(filter(lambda x: x not in df.columns, columns))):
            raise ValueError(f"Columns not found in dataframe: {list(filter(lambda x: x not in df.columns, columns))}")
        columns = list(filter(lambda x: x not in TrackingDataItem.block_columns, columns))
        types = {column: df[column].dtype for column in columns}
        statistics = self.statistics()
        resize_range = TrackingDataItem.resize_range.copy()
        resize_range.update(resize_range_overwrite)
        category_labels: dict[str, None | list] = TrackingDataItem.category_labels.copy()
        category_labels.update(category_labels_overwrite)
        data_map = {}
        for i, column in enumerate(columns):
            if pd.api.types.is_numeric_dtype(types[column]):
                if column in resize_range.keys() and resize_range[column] is not None:
                    vMin, vMax = resize_range[column]
                    df[column] = (df[column] - vMin) / (vMax - vMin)
                else:
                    vMin, vMax = statistics[column]['min'], statistics[column]['max']
                    df[column] = (df[column] - vMin) / (vMax - vMin)
                data_map[column] = {
                    "type": "number",
                    "min": vMin,
                    "max": vMax,
                    "index": i,
                }
            elif pd.api.types.is_string_dtype(types[column]):
                if column in category_labels.keys() and category_labels[column] is not None:
                    label_mapping = {k: v for v, k in enumerate(category_labels[column])}
                    df[column] = df[column].map(label_mapping)
                    data_map[column] = {
                        "type": "category",
                        "labels": category_labels[column],
                        "index": i,
                    }
                else:
                    labels = df[column].unique()
                    if pd.isna(labels).any():
                        labels = labels[1:].tolist()
                    category = pd.Categorical(df[column], categories=labels)
                    df[column] = category.codes
                    data_map[column] = {
                        "type": "category",
                        "labels": labels,
                        "index": i,
                    }
            elif pd.api.types.is_datetime64_any_dtype(types[column]):
                # to timestamp ms
                df[column] = df[column].astype(np.int64) // 10 ** 6
                start = df[column].min()
                end = df[column].max()
                df[column] = ((df[column] - start) / (end - start)) * 10
                data_map[column] = {
                    "type": "datetime",
                    "mode": "timestamp",
                    "index": i,
                }
            else:
                logger.warning(f"Column {column} is not a valid type")
            if not pd.api.types.is_numeric_dtype(df[column].dtype):
                pass
        df = df.fillna(-1)
        df = df[columns]
        tensor = torch.tensor(df.values, dtype=dtype)
        return tensor, data_map
