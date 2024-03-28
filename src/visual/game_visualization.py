import traceback
from datetime import datetime

import numpy as np
import cv2
import pandas as pd
from data import GameNFLData, GameTrackingData, TrackingDataItem, GameFootBallTrackingData


class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        if max([x1, y1, x2, y2]) > 1:
            raise ValueError("Coordinates should be normalized")
        if min([x1, y1, x2, y2]) < 0:
            raise ValueError("Coordinates should be normalized")

    def draw(self, img, color):
        h, w, _ = img.shape
        x1 = int(self.x1 * w)
        x2 = int(self.x2 * w)
        y1 = int(self.y1 * h)
        y2 = int(self.y2 * h)
        cv2.line(img, (x1, y1), (x2, y2), color, 2)

    def __str__(self):
        return f"Line({self.x1}, {self.y1}, {self.x2}, {self.y2})"

    def __repr__(self):
        return str(self)


class Circle:
    def __init__(self, x, y, radius, fill=False):
        self.x = x
        self.y = y
        self.radius = radius
        self.fill = fill
        if max([x, y]) > 1:
            raise ValueError("Coordinates should be normalized")
        if min([x, y]) < 0:
            raise ValueError("Coordinates should be normalized")

    def draw(self, img, color):
        h, w, _ = img.shape
        x = int(self.x * w)
        y = int(self.y * h)
        radius = int(self.radius * w)
        if self.fill:
            cv2.circle(img, (x, y), radius, color, -1)
        else:
            cv2.circle(img, (x, y), radius, color, 2)

    def __str__(self):
        return f"Circle({self.x}, {self.y}, {self.radius})"

    def __repr__(self):
        return str(self)


class Text:
    def __init__(self, x, y, text):
        self.x = x
        self.y = y
        self.text = text
        if max([x, y]) > 1:
            raise ValueError("Coordinates should be normalized")
        if min([x, y]) < 0:
            raise ValueError("Coordinates should be normalized")

    def draw(self, img, color):
        h, w, _ = img.shape
        x = int(self.x * w)
        y = int(self.y * h)
        (text_width, text_height), baseline = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        new_x = x - text_width // 2
        new_y = y + text_height // 2
        cv2.putText(img, self.text, (new_x, new_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def __str__(self):
        return f"Text({self.x}, {self.y}, {self.text})"

    def __repr__(self):
        return str(self)


def get_yard_lines() -> list[Line]:
    lines = []
    for i in range(0, 120, 10):
        lines.append(Line(i / 120, 0, i / 120, 1))
    return lines


def get_players(items: list[TrackingDataItem], cycle_r: float) -> tuple[list[Circle], list[Text]]:
    circles = []
    texts = []
    for item in items:
        circles.append(Circle(item.x / 120, item.y / 53.3, cycle_r, True))
        texts.append(Text(item.x / 120, item.y / 53.3,  f"{item.jerseyNumber: .0f}"))
    return circles, texts


def get_football(item, cycle_r: float) -> Circle:
    football = Circle(item.x / 120, item.y / 53.3, cycle_r, True)
    return football


# def draw_frame(image: np.ndarray, game: GameNFLData, football_tracking: GameFootBallTrackingData, playId, frameId):
#     player = []
#     football = []
#
#     indexes_player = game.tracking.df.index[np.logical_and(game.tracking.df['frameId'] == playId, game.tracking.df['frameId'] == frameId)]
#     for index in indexes_player:
#         player.append(game.tracking[index])
#
#     indexes_football = football_tracking.df.index[football_tracking.df['frameId'] == frameId]
#     for index in indexes_football:
#         football.append(football_tracking[index])
#
#     players, player_texts = get_players(player, 0.01)
#     football = get_football(football[0], 0.01)
#     yard_lines = get_yard_lines()
#     for line in yard_lines:
#         line.draw(image, (255, 255, 255))
#     for player in players:
#         player.draw(image, (0, 0, 255))
#     for player_text in player_texts:
#         player_text.draw(image, (0, 255, 0))
#     football.draw(image, (0, 0, 255))


def draw_by_time(image: np.ndarray, game: GameNFLData, football_tracking: GameFootBallTrackingData, dt: datetime):
    player = []
    football = []

    try:

        indexes_player = game.tracking.df.index[game.tracking.df['time'] == dt]
        for index in indexes_player:
            player.append(game.tracking[index])

        indexes_football = football_tracking.df.index[football_tracking.df['time'] == dt]
        for index in indexes_football:
            football.append(football_tracking[index])

        players, player_texts = get_players(player, 0.015)
        football = get_football(football[0], 0.005)
        yard_lines = get_yard_lines()
        for line in yard_lines:
            line.draw(image, (127, 127, 127))
        for player in players:
            player.draw(image, (0, 0, 255))
        for player_text in player_texts:
            player_text.draw(image, (0, 255, 0))
        football.draw(image, (255, 0, 0))

    except:
        traceback.print_exc()
        pass


# def draw(image: np.ndarray, game: GameNFLData, football: GameFootBallTrackingData, playId):
#     frameIds = game.df['frameId'].unique()
#     for frameId in frameIds:
#         draw_frame(image, game, football, playId, frameId)
