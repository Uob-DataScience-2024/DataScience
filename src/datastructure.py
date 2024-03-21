import re
from datetime import datetime

import pandas as pd
from playdata import PlayDataItem, GamePlayData
from pffdata import PffDataItem, GamePffData
from trackingdata import TrackingDataItem, GameTrackingData


class NFLDataItem:
    # TrackingData Part
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
    # PffData Part
    pff_role: str
    pff_positionLinedUp: str
    pff_blockType: str

    pff_hit: bool
    pff_hurry: bool
    pff_sack: bool
    pff_beatenByDefender: bool
    pff_hitAllowed: bool
    pff_hurryAllowed: bool
    pff_sackAllowed: bool

    pff_nflIdBlockedPlayer: int
    # PlayData Part

    quarter: int
    down: int
    yardsToGo: int
    yardlineNumber: int
    preSnapHomeScore: int
    preSnapVisitorScore: int
    penaltyYards: int
    prePenaltyPlayResult: int
    playResult: int
    foulNFLId1: int
    foulNFLId2: int
    foulNFLId3: int
    absoluteYardlineNumber: int
    defendersInTheBox: int

    pff_playAction: bool

    playDescription: str
    possessionTeam: str
    defensiveTeam: str
    yardlineSide: str
    passResult: str
    foulName1: str
    foulName2: str
    foulName3: str
    offenseFormation: str
    personnelO: str
    personnelD: str
    pff_passCoverage: str
    pff_passCoverageType: str

    no_payload_columns = {
        'game_id': 'gameId',
        'play_id': 'playId',
        'nfl_id': 'nflId',
        'frame_id': 'frameId',
        'dt': 'time',
    }

    binary_list = [
        'pff_hit',
        'pff_hurry',
        'pff_sack',
        'pff_beatenByDefender',
        'pff_hitAllowed',
        'pff_hurryAllowed',
        'pff_sackAllowed',
        'pff_playAction',
    ]

    def __init__(self, week: str, gameId: int, playId: int, nflId: int, frameId: int, dt: datetime, number_payload: dict, binary_payload: dict, text_payload: dict):
        self.week = week
        self.gameId = gameId
        self.playId = playId
        self.nflId = nflId
        self.frameId = frameId
        self.time = dt
        for key, value in number_payload.items():
            setattr(self, key, value)
        for key, value in binary_payload.items():
            setattr(self, key, value)
        for key, value in text_payload.items():
            setattr(self, key, value)
        self.number_payload = number_payload
        self.binary_payload = binary_payload
        self.text_payload = text_payload

        for key in self.binary_list:
            if type(getattr(self, key)) == float:
                if getattr(self, key) == 1.0:
                    setattr(self, key, True)
                elif getattr(self, key) == 0.0:
                    setattr(self, key, False)

    @staticmethod
    def from_object(tracking: TrackingDataItem, pff: PffDataItem, play: PlayDataItem):
        number_payload = tracking.number_payload
        number_payload.update(play.number_payload)
        number_payload.update(pff.number_payload)

        binary_payload = play.binary_payload
        binary_payload.update(pff.binary_payload)

        text_payload = tracking.text_payload
        text_payload.update(play.text_payload)
        text_payload.update(pff.text_payload)

        return NFLDataItem(tracking.week, tracking.game_id, tracking.play_id, tracking.nfl_id, tracking.frame_id, tracking.time, number_payload, binary_payload, text_payload)

    def __str__(self) -> str:
        return f"NFLDataItem(week={self.week}, gameId={self.gameId}, playId={self.playId}, nflId={self.nflId}, frameId={self.frameId}, time={self.time})"


class GameNFLData:
    def __init__(self, gameId: int, df_tracking: pd.DataFrame, df_pff: pd.DataFrame, df_play: pd.DataFrame, week: str = ''):
        self.gameId = gameId
        self.date_start = df_tracking['time'].min()
        self.date_end = df_tracking['time'].max()
        self.tracking = GameTrackingData(gameId, self.date_start, self.date_end, week, df_tracking)
        self.pff = GamePffData(gameId, df_pff)
        self.play = GamePlayData(gameId, df_play)
        self.df = self.merge()

    def merge(self) -> pd.DataFrame:
        df_tracking = self.tracking.df.copy()
        df_pff = self.pff.df.copy()
        df_play = self.play.df.copy()
        df_tracking = df_tracking.dropna(subset=['nflId', 'playId'])
        df_pff = df_pff.dropna(subset=['nflId', 'playId'])
        df_play = df_play.dropna(subset=['playId'])
        df_tracking['union_id'] = df_tracking['nflId'].astype(str) + df_tracking['playId'].astype(str)
        df_pff['union_id'] = df_pff['nflId'].astype(str) + df_pff['playId'].astype(str)
        df_pff = df_pff.drop(columns=['nflId', 'playId', 'gameId'])
        df_play = df_play.drop(columns=['gameId'])

        result = pd.merge(df_tracking, df_pff, on='union_id', how='left')
        result = pd.merge(result, df_play, on='playId', how='left')
        return result

    @staticmethod
    def load(filename_tracking, filename_pff, filename_play) -> dict:
        week = re.search(r'week(\d+)', filename_tracking).group(1)
        week = str(int(week))
        df_tracking = pd.read_csv(filename_tracking)
        df_tracking['time'] = pd.to_datetime(df_tracking['time'])
        df_pff = pd.read_csv(filename_pff)
        df_play = pd.read_csv(filename_play)
        loaded = {}
        for gameId in df_tracking['gameId'].unique():
            sub_df_tracking = df_tracking[df_tracking['gameId'] == gameId]
            sub_df_pff = df_pff[df_pff['gameId'] == gameId]
            sub_df_play = df_play[df_play['gameId'] == gameId]
            loaded[gameId] = GameNFLData(gameId, sub_df_tracking, sub_df_pff, sub_df_play, week)
        return loaded

    @staticmethod
    def loads(filename_tracking_list, filename_pff, filename_play) -> dict:
        df_pff = pd.read_csv(filename_pff)
        df_play = pd.read_csv(filename_play)
        preload = {}
        for gameId in df_pff['gameId'].unique():
            sub_df_pff = df_pff[df_pff['gameId'] == gameId]
            sub_df_play = df_play[df_play['gameId'] == gameId]
            preload[gameId] = (sub_df_pff, sub_df_play)
        loaded = {}
        for filename_tracking in filename_tracking_list:
            week = re.search(r'week(\d+)', filename_tracking).group(1)
            week = str(int(week))
            df_tracking = pd.read_csv(filename_tracking)
            df_tracking['time'] = pd.to_datetime(df_tracking['time'])
            for gameId in df_tracking['gameId'].unique():
                sub_df_tracking = df_tracking[df_tracking['gameId'] == gameId]
                sub_df_pff, sub_df_play = preload[gameId]
                loaded[gameId] = GameNFLData(gameId, sub_df_tracking, sub_df_pff, sub_df_play, week)
        return loaded

    def __str__(self) -> str:
        return f"GameNFLData(gameId={self.gameId}, date_start={self.date_start}, date_end={self.date_end}, tracking={self.tracking}, pff={self.pff}, play={self.play})"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> NFLDataItem:
        row = self.df.iloc[idx]
        tracking_row = row[self.tracking.df.columns]
        pff_row = row[self.pff.df.columns]
        play_row = row[self.play.df.columns]
        tracking_args = {arg_name: tracking_row[col_name] for arg_name, col_name in self.tracking.no_payload_columns}
        tracking_args['number_payload'] = {col_name: tracking_row[col_name] for col_name, dtype in self.tracking.number_list}
        tracking_args['text_payload'] = {col_name: tracking_row[col_name] for col_name, dtype in self.tracking.text_list}

        pff_args = {arg_name: pff_row[col_name] for arg_name, col_name in self.pff.no_payload_columns}
        pff_args['number_payload'] = {col_name: pff_row[col_name] for col_name, dtype in self.pff.number_list}
        pff_args['binary_payload'] = {col_name: pff_row[col_name] for col_name, dtype in self.pff.binary_category_list}
        pff_args['text_payload'] = {col_name: pff_row[col_name] for col_name, dtype in self.pff.text_list}

        play_args = {arg_name: play_row[col_name] for arg_name, col_name in self.play.no_payload_columns}
        play_args['number_payload'] = {col_name: play_row[col_name] for col_name, dtype in self.play.number_list}
        play_args['binary_payload'] = {col_name: play_row[col_name] for col_name, dtype in self.play.binary_category_list}
        play_args['text_payload'] = {col_name: play_row[col_name] for col_name, dtype in self.play.text_list}

        return NFLDataItem.from_object(TrackingDataItem(self.tracking.week, **tracking_args), PffDataItem(**pff_args), PlayDataItem(**play_args))
