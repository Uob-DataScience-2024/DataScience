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

    def adjust_color(self, color, factor):
        """ 调整颜色亮度 """
        return tuple(np.clip(np.array(color) * factor, 0, 255).astype(int))

    def draw(self, img, color):
        h, w, _ = img.shape
        x = int(self.x * w)
        y = int(self.y * h)
        radius = int(self.radius * w)

        if self.fill:
            # 绘制边框（更大的圆）
            border_radius = int(radius * 1.2)  # 边框圆的半径
            cv2.circle(img, (x, y), border_radius, (202, 195, 232), -1)
            # start_color = self.adjust_color(color, 0.5)  # 开始颜色
            # end_color = self.adjust_color(color, 1.5)  # 结束颜色
            # for i in range(border_radius, radius, -1):
            #     inter_factor = (i - radius) / (border_radius - radius)
            #     inter_color1 = (start_color[0] * inter_factor, start_color[1] * inter_factor, start_color[2] * inter_factor)
            #     inter_color2 = (end_color[0] * (1 - inter_factor), end_color[1] * (1 - inter_factor), end_color[2] * (1 - inter_factor))
            #     inter_color = inter_color1[0] + inter_color2[0], inter_color1[1] + inter_color2[1], inter_color1[2] + inter_color2[2]
            #     cv2.circle(img, (x, y), i, inter_color, 1)

            # 绘制内圆
            cv2.circle(img, (x, y), radius, color, -1)
        else:
            cv2.circle(img, (x, y), radius, color, 2)

    # def draw(self, img, color):
    #     h, w, _ = img.shape
    #     x = int(self.x * w)
    #     y = int(self.y * h)
    #     radius = int(self.radius * w)
    #     if self.fill:
    #         cv2.circle(img, (x, y), radius, color, -1)
    #     else:
    #         cv2.circle(img, (x, y), radius, color, 2)

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


def get_players(items: list[TrackingDataItem], cycle_r: float, home_team: str) -> tuple[list[Circle], list[Text], list[bool]]:
    circles = []
    texts = []
    is_home = []
    for item in filter(lambda x: not pd.isna(x.nfl_id), items):
        is_home.append(item.team == home_team)
        circles.append(Circle(item.x / 120, item.y / 53.3, cycle_r, True))
        texts.append(Text(item.x / 120, item.y / 53.3, f" {item.jerseyNumber:.0f} "))
    return circles, texts, is_home


def get_football(item, cycle_r: float) -> Circle:
    football = Circle(item.x / 120, item.y / 53.3, cycle_r, True)
    return football


def draw_by_time(image: np.ndarray, game: GameNFLData, football_tracking: GameFootBallTrackingData, dt: datetime, home_team: str):
    player = []
    football = []

    try:

        indexes_player = game.tracking.df.index[game.tracking.df['time'] == dt]
        for index in indexes_player:
            player.append(game.tracking[index])

        indexes_football = football_tracking.df.index[football_tracking.df['time'] == dt]
        for index in indexes_football:
            football.append(football_tracking[index])

        players, player_texts, is_home = get_players(player, 0.015, home_team)
        football = get_football(football[0], 0.005)
        yard_lines = get_yard_lines()
        for line in yard_lines:
            line.draw(image, (172, 88, 245))
        for i, (player, player_text) in enumerate(zip(players, player_texts)):
            player.draw(image, (14, 165, 233) if is_home[i] else (255, 80, 110))
            player_text.draw(image, (246, 227, 93))
        football.draw(image, (55, 71, 108))

    except:
        traceback.print_exc()
        pass


def draw_by_time_df(image: np.ndarray, df: pd.DataFrame, dt: datetime, home_team: str):
    cycle_r = 0.015
    cycle_r_football = 0.005
    try:

        player = df[np.logical_and(df['time'] == dt, ~np.isnan(df['nflId']))]
        # for index in indexes_player:
        #     player.append(game.tracking[index])

        football = df[np.logical_and(df['time'] == dt, np.isnan(df['nflId']))].iloc[0]

        circles = []
        texts = []
        is_home = []
        for i, item in player.iterrows():
            is_home.append(item['team'] == home_team)
            circles.append(Circle(item['x'] / 120, item['y'] / 53.3, cycle_r, True))
            texts.append(Text(item.x / 120, item.y / 53.3, f" {item.jerseyNumber:.0f} "))
        football = Circle(football['x'] / 120, football['y'] / 53.3, cycle_r_football, True)
        yard_lines = get_yard_lines()
        for line in yard_lines:
            line.draw(image, (172, 88, 245))
        for i, (player, player_text) in enumerate(zip(circles, texts)):
            player.draw(image, (14, 165, 233) if is_home[i] else (255, 80, 110))
            player_text.draw(image, (246, 227, 93))
        football.draw(image, (55, 71, 108))

    except:
        traceback.print_exc()
        pass


def draw_background(image: np.ndarray, banner="", color: tuple[int, int, int] = (255, 255, 255), font_color: tuple[int, int, int] = (0, 0, 0),
                    amplification_factor_x=1.3,
                    amplification_factor_y_top=1.2,
                    amplification_factor_y_bottom=1.4):
    h, w = image.shape[:2]
    top = int((amplification_factor_y_top - 1) * h)
    bottom = int((amplification_factor_y_bottom - 1) * h)
    new_h, new_w = h + top + bottom, int(w * amplification_factor_x)
    # copy the image to the center of the new image
    new_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    new_image[:, :] = color
    x_offset = (new_w - w) // 2
    y_offset = top
    new_image[y_offset:y_offset + h, x_offset:x_offset + w] = image
    if banner != "":
        x, y = (new_w // 2, y_offset // 4)
        (text_width, text_height), baseline = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
        x = x - text_width // 2
        y = y + text_height // 2
        cv2.putText(new_image, banner, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, font_color, 4)
    return new_image, y_offset + h


def draw_status(image: np.ndarray, game: GameNFLData, dt: datetime, home_team: str, start_y: int,
                font_size=1.5, thickness=2,
                jerseyNumberColor=(13, 148, 136), text_color=(219, 63, 41)):
    h, w = image.shape[:2]
    nfls = []
    # indexes_player = game.tracking.df.index[game.tracking.df['time'] == dt]
    indexes_player = game.df.index[game.df['time'] == dt]
    for index in indexes_player:
        nfls.append(game[index])

    home_data = []
    visitor_data = []
    for nfl in nfls:
        if nfl.team == home_team:
            home_data.append(nfl)
        else:
            visitor_data.append(nfl)

    # cv2.line(image, (0, start_y), (w, start_y), (0, 0, 0), 2)
    cv2.line(image, (w // 2, int(start_y + (h - start_y) * 0.2)), (w // 2, int(start_y + (h - start_y) * 0.8)), (65, 203, 191), 4)

    block_top_margin = 0.005
    block_bottom_margin = 0.005
    block_left_margin = 0.005
    block_right_margin = 0.005
    x_base = 0.025 * w
    y_base = start_y + (h - start_y) * 0.3
    draw_player_info(block_bottom_margin, block_left_margin, block_right_margin, block_top_margin, font_size, h, home_data, image, jerseyNumberColor, text_color, thickness, w, x_base, y_base, w // 2)
    x_base += w // 2
    draw_player_info(block_bottom_margin, block_left_margin, block_right_margin, block_top_margin, font_size, h, visitor_data, image, jerseyNumberColor, text_color, thickness, w, x_base, y_base,
                     w * 0.98)


def draw_player_info(block_bottom_margin, block_left_margin, block_right_margin, block_top_margin, font_size, h, home_data, image, jerseyNumberColor, text_color, thickness, w, x_base, y_base, x_end):
    current_x = 0
    current_y = 0
    for i, nfl in enumerate(home_data):
        text = f"{nfl.jerseyNumber:.0f}: {nfl.pff_role}"
        text_p1 = f"{nfl.jerseyNumber:.0f}: "
        text_p2 = f"{nfl.pff_role}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)
        (text_width_p1, text_height_p1), baseline_p1 = cv2.getTextSize(text_p1, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)
        block_width = text_width + block_left_margin * w + block_right_margin * w
        block_height = text_height + block_top_margin * h + block_bottom_margin * h
        if x_base + current_x + block_width > x_end:
            current_x = 0
            current_y += block_height
        x = int(x_base + current_x + block_left_margin * w)
        y = int(y_base + current_y + block_top_margin * h + text_height / 2)
        cv2.putText(image, text_p1, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, jerseyNumberColor, thickness)
        cv2.putText(image, text_p2, (x + text_width_p1, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, thickness)
        current_x += block_width


def draw_status_df(image: np.ndarray, df: pd.DataFrame, dt: datetime, home_team: str, start_y: int,
                   font_size=1.5, thickness=2,
                   jerseyNumberColor=(13, 148, 136), text_color=(219, 63, 41), targetX='jerseyNumber', targetY='pff_role', draw_once=False):
    h, w = image.shape[:2]
    player = df[df['time'] == dt]
    home_data = player[player['team'] == home_team]
    visitor_data = player[player['team'] != home_team]

    if not draw_once:
        cv2.line(image, (w // 2, int(start_y + (h - start_y) * 0.2)), (w // 2, int(start_y + (h - start_y) * 0.8)), (65, 203, 191), 4)

        block_top_margin = 0.005
        block_bottom_margin = 0.005
        block_left_margin = 0.005
        block_right_margin = 0.005
        x_base = 0.025 * w
        y_base = start_y + (h - start_y) * 0.3
        draw_player_info_df(block_bottom_margin, block_left_margin, block_right_margin, block_top_margin, font_size, h, home_data, image, jerseyNumberColor, text_color, thickness, w, x_base, y_base,
                            w // 2, targetX, targetY)
        x_base += w // 2
        draw_player_info_df(block_bottom_margin, block_left_margin, block_right_margin, block_top_margin, font_size, h, visitor_data, image, jerseyNumberColor, text_color, thickness, w, x_base,
                            y_base,
                            w * 0.98, targetX, targetY)
    else:
        center_x = w // 2
        center_y = start_y + (h - start_y) * 0.5
        d = player.iloc[0]
        text = f"{d[targetX]}: {d[targetY]}"
        text_p1 = f"{d[targetX]}: "
        text_p2 = f"{d[targetY]}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)
        (text_width_p1, text_height_p1), baseline_p1 = cv2.getTextSize(text_p1, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)
        x = int(center_x - text_width // 2)
        y = int(center_y + text_height // 2)
        cv2.putText(image, text_p1, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, jerseyNumberColor, thickness)
        cv2.putText(image, text_p2, (x + text_width_p1, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, thickness)


def draw_player_info_df(block_bottom_margin, block_left_margin, block_right_margin, block_top_margin, font_size, h, data, image, jerseyNumberColor, text_color, thickness, w, x_base, y_base,
                        x_end, targetX='jerseyNumber', targetY='pff_role'):
    current_x = 0
    current_y = 0
    for i, nfl in data.iterrows():
        if pd.isna(nfl[targetX]) or pd.isna(nfl[targetY]):
            continue
        text = f"{nfl[targetX]:.0f}: {nfl[targetY]}"
        text_p1 = f"{nfl[targetX]:.0f}: "
        text_p2 = f"{nfl[targetY]}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)
        (text_width_p1, text_height_p1), baseline_p1 = cv2.getTextSize(text_p1, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)
        block_width = text_width + block_left_margin * w + block_right_margin * w
        block_height = text_height + block_top_margin * h + block_bottom_margin * h
        if x_base + current_x + block_width > x_end:
            current_x = 0
            current_y += block_height
        x = int(x_base + current_x + block_left_margin * w)
        y = int(y_base + current_y + block_top_margin * h + text_height / 2)
        cv2.putText(image, text_p1, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, jerseyNumberColor, thickness)
        cv2.putText(image, text_p2, (x + text_width_p1, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, thickness)
        current_x += block_width


def draw_play_info_by_template(image: np.ndarray, df: pd.DataFrame, dt: datetime,
                               start_y: int, font_size=1.5, thickness=2, jerseyNumberColor=(13, 148, 136), text_color=(219, 63, 41),
                               target_columns=[{'col': 'passResult', 'template': '{key}: {value}'}]):
    h, w = image.shape[:2]
    data = df[df['time'] == dt]

    block_top_margin = 0.005
    block_bottom_margin = 0.005
    block_left_margin = 0.005
    block_right_margin = 0.005
    x_base = 0.025 * w
    y_base = start_y + (h - start_y) * 0.3

    current_x = 0
    current_y = 0
    for info in target_columns:
        col = info['col']
        template = info['template']
        sub_data = data[col]
        sub_data = sub_data.unique()
        value = sub_data[0]
        text = template.format(key=col, value=value)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)
        block_width = text_width + block_left_margin * w + block_right_margin * w
        block_height = text_height + block_top_margin * h + block_bottom_margin * h
        if x_base + current_x + block_width > w:
            current_x = 0
            current_y += block_height
        x = int(x_base + current_x + block_left_margin * w)
        y = int(y_base + current_y + block_top_margin * h + text_height / 2)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, jerseyNumberColor, thickness)
        current_x += block_width

