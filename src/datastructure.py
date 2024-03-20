from datetime import datetime


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
        for key, value in number_payload.items():
            setattr(self, key, value)
        for key, value in text_payload.items():
            setattr(self, key, value)
