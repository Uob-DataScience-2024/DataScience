import re
from datetime import datetime

import pandas as pd


class PffDataItem:
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

    no_payload_columns = {
        'gameId': 'gameId',
        'playId': 'playId',
        'nflId': 'nflId',
    }

    binary_list = [
        'pff_hit',
        'pff_hurry',
        'pff_sack',
        'pff_beatenByDefender',
        'pff_hitAllowed',
        'pff_hurryAllowed',
        'pff_sackAllowed',
    ]

    def __init__(self, gameId: int, playId: int, nflId: int, number_payload: dict, binary_category_payload, text_payload: dict):
        self.gameId = gameId
        self.playId = playId
        self.nflId = nflId
        for key, value in number_payload.items():
            setattr(self, key, value)
        for key, value in text_payload.items():
            setattr(self, key, value)
        for key, value in binary_category_payload.items():
            setattr(self, key, value)
        for key in self.binary_list:
            if type(getattr(self, key)) == float:
                if getattr(self, key) == 1.0:
                    setattr(self, key, True)
                elif getattr(self, key) == 0.0:
                    setattr(self, key, False)
