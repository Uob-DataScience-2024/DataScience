import re
from datetime import datetime

import pandas as pd


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

    event: str
    """Event that occurred(not each line has an event)"""

    no_payload_columns = {
        'game_id': 'gameId',
        'play_id': 'playId',
        'nfl_id': 'nflId',
        'frame_id': 'frameId',
        'dt': 'time',
    }

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
