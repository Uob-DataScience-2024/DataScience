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

    def __init__(self, week: str, gameId: int, playId: int, nflId: int, number_payload: dict, binary_category_payload, text_payload: dict):
        self.week = week
        self.gameId = gameId
        self.playId = playId
        self.nflId = nflId
        for key, value in number_payload.items():
            setattr(self, key, value)
        for key, value in text_payload.items():
            setattr(self, key, value)
        for key, value in binary_category_payload.items():
            setattr(self, key, value)
