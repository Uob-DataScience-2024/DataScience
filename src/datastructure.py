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
