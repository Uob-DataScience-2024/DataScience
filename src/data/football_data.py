import re
from datetime import datetime

import pandas as pd

from . import GameTrackingData


class GameFootBallTrackingData(GameTrackingData):
    def __init__(self, game_id: int, date_start: datetime, date_end: datetime, week: str, df: pd.DataFrame):
        super().__init__(game_id, date_start, date_end, week, df)

    @staticmethod
    def load(filename):
        week = re.search(r'week(\d+)', filename).group(1)
        week = str(int(week))
        df = pd.read_csv(filename)
        df['time'] = pd.to_datetime(df['time'])
        loaded = {}
        for game_id in df['gameId'].unique():
            sub_df = df[df['gameId'] == game_id]
            sub_df = sub_df[pd.isna(sub_df['nflId'])]
            date_start = sub_df['time'].min()
            date_end = sub_df['time'].max()
            sub_df = sub_df.reset_index(drop=True)
            loaded[game_id] = GameTrackingData(game_id, date_start, date_end, week, sub_df)
        return loaded

    @staticmethod
    def load_all(filenames):
        loaded = {}
        for filename in filenames:
            loaded.update(GameTrackingData.load(filename))
        return loaded

    @staticmethod
    def from_tracking_data(tracking_data: GameTrackingData):
        df = tracking_data.df[pd.isna(tracking_data.df['nflId'])]
        date_start = df['time'].min()
        date_end = df['time'].max()
        df = df.reset_index(drop=True)
        return GameTrackingData(tracking_data.game_id, date_start, date_end, tracking_data.week, df)
