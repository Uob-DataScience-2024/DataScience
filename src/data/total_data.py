import enum
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import pandas as pd


class SplitMode(enum.Enum):
    Game = 1
    PlayID = 2
    FrameID = 3

    Time = 4
    TimeStage = 5

    NflID = 6


class TrackingNormData:
    def __init__(self, path_list: list, df: pd.DataFrame = None, parallel_loading: bool = True):
        self.path_list = path_list
        self.data: pd.DataFrame = pd.DataFrame() if df is None else df
        if df is None:
            self.load_data(parallel_loading)

    def load_data(self, parallel_loading: bool = False):
        if parallel_loading:
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = executor.map(pd.read_csv, self.path_list)
                self.data = pd.concat(results)
        else:
            for path in self.path_list:
                self.data = pd.concat([self.data, pd.read_csv(path)])
        self.data['time'] = pd.to_datetime(self.data['time'], format='mixed')
        self.data.sort_values(by=['gameId', 'playId', 'frameId', 'time'], inplace=True)

    def split(self, mode: SplitMode):
        df = self.data.copy()
        if mode == SplitMode.Game:
            df = df.groupby('gameId')
        elif mode == SplitMode.PlayID:
            df = df.groupby(['gameId', 'playId'])
        elif mode == SplitMode.FrameID:
            df = df.groupby(['gameId', 'playId', 'frameId'])
        elif mode == SplitMode.Time:
            df = df.groupby(['time'])
        elif mode == SplitMode.TimeStage:
            df['time_diff'] = df['time'].diff().dt.total_seconds()
            threshold = 10
            df['group'] = (df['time_diff'] > threshold).cumsum()
            df = df.groupby('group')
        elif mode == SplitMode.NflID:
            df = df.groupby('nflId')
        return TrackingNormData(self.path_list, {name: group.copy() for name, group in df})


class PffNormData:
    def __init__(self, path: str, df: pd.DataFrame = None):
        self.path = path
        self.data: pd.DataFrame = pd.DataFrame() if df is None else df
        if df is None:
            self.load_data()

    def load_data(self):
        self.data = pd.read_csv(self.path)
        self.data.sort_values(by=['gameId', 'playId', 'nflId'], inplace=True)

    def split(self, mode: SplitMode):
        df = self.data.copy()
        if mode == SplitMode.Game:
            df = df.groupby('gameId')
        elif mode == SplitMode.PlayID:
            df = df.groupby(['gameId', 'playId'])
        elif mode == SplitMode.FrameID:
            # df = df.groupby(['gameId', 'playId', 'frameId'])
            raise NotImplementedError
        elif mode == SplitMode.Time:
            df = df.groupby(['time'])
        elif mode == SplitMode.TimeStage:
            df['time_diff'] = df['time'].diff().dt.total_seconds()
            threshold = 10
            df['group'] = (df['time_diff'] > threshold).cumsum()
            df = df.groupby('group')
        elif mode == SplitMode.NflID:
            df = df.groupby('nflId')
        return PffNormData(self.path, {name: group.copy() for name, group in df})


class PlayNormData:
    def __init__(self, path: str, df: pd.DataFrame = None):
        self.path = path
        self.data: pd.DataFrame = pd.DataFrame() if df is None else df
        if df is None:
            self.load_data()

    def load_data(self):
        self.data = pd.read_csv(self.path)
        self.data.sort_values(by=['gameId', 'playId'], inplace=True)

    def split(self, mode: SplitMode):
        df = self.data.copy()
        if mode == SplitMode.Game:
            df = df.groupby('gameId')
        elif mode == SplitMode.PlayID:
            df = df.groupby(['gameId', 'playId'])
        elif mode == SplitMode.FrameID:
            raise NotImplementedError
        elif mode == SplitMode.Time:
            df = df.groupby(['time'])
        elif mode == SplitMode.TimeStage:
            df['time_diff'] = df['time'].diff().dt.total_seconds()
            threshold = 10
            df['group'] = (df['time_diff'] > threshold).cumsum()
            df = df.groupby('group')
        elif mode == SplitMode.NflID:
            raise NotImplementedError
        return PlayNormData(self.path, {name: group.copy() for name, group in df})


class PlayerNormData:
    def __init__(self, path: str, df: pd.DataFrame = None):
        self.path = path
        self.data: pd.DataFrame = pd.DataFrame() if df is None else df
        if df is None:
            self.load_data()

    def load_data(self):
        self.data = pd.read_csv(self.path)
        self.data.sort_values(by=['nflId'], inplace=True)

    def split(self, mode: SplitMode):
        raise NotImplementedError


class GameNormData:
    def __init__(self, path: str, df: pd.DataFrame = None):
        self.path = path
        self.data: pd.DataFrame = pd.DataFrame() if df is None else df
        if df is None:
            self.load_data()

    def load_data(self):
        self.data = pd.read_csv(self.path)
        self.data.sort_values(by=['gameId'], inplace=True)

    def split(self, mode: SplitMode):
        raise NotImplementedError


class MergeNormData:
    def __init__(self, player: PlayerNormData, game: GameNormData, tracking: TrackingNormData, pff: PffNormData = None, play: PlayNormData = None):
        self.game = tracking.data.copy()
        self.game = self.game.merge(pff.data, on=['gameId', 'playId', 'nflId'], how='left') if pff is not None else self.game
        self.game = self.game.merge(play.data, on=['gameId', 'playId'], how='left') if play is not None else self.game
        self.player = player.data.copy()
        self.game_info = game.data.copy()

    def split(self, mode: SplitMode):
        df = self.game.copy()
        if mode == SplitMode.Game:
            df = df.groupby('gameId')
        elif mode == SplitMode.PlayID:
            df = df.groupby(['gameId', 'playId'])
        elif mode == SplitMode.FrameID:
            df = df.groupby(['gameId', 'playId', 'frameId'])
        elif mode == SplitMode.Time:
            df = df.groupby(['time'])
        elif mode == SplitMode.TimeStage:
            df['time_diff'] = df['time'].diff().dt.total_seconds()
            threshold = 10
            df['group'] = (df['time_diff'] > threshold).cumsum()
            df = df.groupby('group')
        elif mode == SplitMode.NflID:
            df = df.groupby('nflId')
        return MergeNormData(PlayerNormData("", self.player), GameNormData("", self.game_info), TrackingNormData([], {name: group.copy() for name, group in df}), None, None)


def preprocess(merge: MergeNormData):
    # new col: according to passResult mark the defendersInBox, personelD, pff_passCoverage, pff_passCoverageType
    pass
