import threading

from visual.game_visualization import *
from data import GameNFLData, GameTrackingData, TrackingDataItem, GameFootBallTrackingData, GameData
from data.total_data import TrackingNormData, PffNormData, PlayNormData, PlayerNormData, GameNormData, MergeNormData
from loguru import logger
from matplotlib import pyplot as plt
from tqdm.rich import tqdm
import argparse
from rich.console import Console

import os
import subprocess
import unittest
from datetime import timedelta


class VideoPipline:
    def __init__(self, encoder, output, fps):
        self.encoder = encoder
        self.output = output
        self.fps = fps
        self.inited = False
        self.process = None
        self.out = None

    def build(self, width, height):
        if self.encoder == "ffmpeg":
            command = [
                'ffmpeg',
                '-y',  # 覆盖输出文件
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',  # 分辨率
                '-pix_fmt', 'bgr24',  # OpenCV的默认格式
                '-r', f'{self.fps}',  # 帧率
                '-i', '-',  # 输入来自stdin
                '-c:v', 'h264_nvenc',  # 使用NVIDIA的h264编码器
                '-pix_fmt', 'yuv420p',  # 输出像素格式
                f'{self.output}'  # 输出文件名
            ]
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(f'{self.output}', fourcc, self.fps, (width, height))
        self.inited = True

    def write(self, frame):
        if self.encoder == "ffmpeg":
            self.process.stdin.write(frame.tobytes())
        else:
            self.out.write(frame)

    def close(self):
        if self.encoder == "ffmpeg":
            self.process.stdin.close()
            self.process.wait()
            if self.process.returncode != 0:
                raise RuntimeError('FFmpeg ERROR，Code {}'.format(self.process.returncode))
        else:
            self.out.release()


class Visual:
    def __init__(self, args):
        if args is None:
            return
        self.data_dir = args.data_dir
        self.fps = args.fps
        self.encoder = args.encoder
        self.output = args.output
        self.gameid = args.gameid
        self.time_max = args.time_max

    def run(self):
        games, game_data = self.load_demo_data_nfl_data()
        if self.gameid not in games:
            raise ValueError(f"Game {self.gameid} not found")
        game = games[self.gameid]
        football = GameFootBallTrackingData.from_tracking_data(game.tracking)
        home_visitor = game_data.get_home_visitor()
        dts = game.tracking.df['time'].unique()
        dts = list(sorted(dts))
        start = dts[0]
        if self.time_max > 0:
            dts = [x for x in dts if x <= start + timedelta(seconds=self.time_max)]
        result = []
        last_dt = dts[0]
        for dt in tqdm(dts):
            image = np.zeros((1106, 2400, 3), dtype=np.uint8)
            image.fill(255)
            draw_by_time(image, game, football, dt, home_visitor[game.gameId][0])
            ms_interval = (dt - last_dt).total_seconds() * 1000
            if ms_interval > 1000 * 5:
                ms_interval = 1000
            last_dt = dt
            # draw datetime
            image, bottom_y = draw_background(image, f"{home_visitor[game.gameId][0]} vs {home_visitor[game.gameId][1]} - {str(game.gameId)}", (209, 222, 233), (242, 149, 89))
            draw_status(image, game, dt, home_visitor[game.gameId][0], bottom_y)
            cv2.putText(image, dt.strftime('%Y-%m-%d %H:%M:%S'), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (32, 44, 57), 2)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result.append((np.array(image, dtype=np.uint8), ms_interval))

        if self.encoder == "ffmpeg":
            self.ffmpeg_encode(result)
        else:
            self.opencv_encode(result)

    # def run_new(self, gameid, tracking: TrackingNormData, pff: PffNormData, play: PlayNormData, game: GameNormData, player: PlayerNormData, merge: MergeNormData, targetX='jerseyNumber', targetY='pff_role'):
    #     df = merge.game.copy()
    #     df = df[df['gameId'] == gameid]
    #     home_visitor = merge.game_info[merge.game_info['gameId'] == game][['homeTeamAbbr', 'visitorTeamAbbr']].values[0]
    #     dts = df['time'].unique()
    #     start = dts[0]
    #     if self.time_max > 0:
    #         dts = [x for x in dts if x <= start + timedelta(seconds=self.time_max)]
    #     result = []
    #     last_dt = dts[0]
    #     for dt in tqdm(dts):
    #         image = np.zeros((1106, 2400, 3), dtype=np.uint8)
    #         image.fill(255)
    #         draw_by_time_df(image, df, dt, home_visitor)
    #         ms_interval = (dt - last_dt).total_seconds() * 1000
    #         if ms_interval > 1000 * 5:
    #             ms_interval = 1000
    #         last_dt = dt
    #         # draw datetime
    #         image, bottom_y = draw_background(image, f"{home_visitor[0]} vs {home_visitor[1]} - {str(game)}", (209, 222, 233), (242, 149, 89))
    #         draw_status_df(image, df, dt, home_visitor[0], bottom_y, targetY='event')
    #         cv2.putText(image, dt.strftime('%Y-%m-%d %H:%M:%S'), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (32, 44, 57), 2)
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         result.append((np.array(image, dtype=np.uint8), ms_interval))
    #         yield image.copy()
    #
    #     if self.encoder == "ffmpeg":
    #         self.ffmpeg_encode(result)
    #     else:
    #         self.opencv_encode(result)

    def run_low_memory(self):
        games, game_data = self.load_demo_data_nfl_data()
        if self.gameid not in games:
            raise ValueError(f"Game {self.gameid} not found")
        game = games[self.gameid]
        football = GameFootBallTrackingData.from_tracking_data(game.tracking)
        home_visitor = game_data.get_home_visitor()
        dts = game.tracking.df['time'].unique()
        dts = list(sorted(dts))
        start = dts[0]
        if self.time_max > 0:
            dts = [x for x in dts if x <= start + timedelta(seconds=self.time_max)]
        pipline = VideoPipline(self.encoder, self.output, self.fps)
        one_frame_duration_ms = 1000 / self.fps
        last_dt = dts[0]
        for dt in tqdm(dts) if not self.encoder == "ffmpeg" else dts:
            image = np.zeros((1106, 2400, 3), dtype=np.uint8)
            image.fill(255)
            draw_by_time(image, game, football, dt, home_visitor[game.gameId][0])
            ms_interval = (dt - last_dt).total_seconds() * 1000
            if ms_interval > 1000 * 5:
                ms_interval = 1000
            last_dt = dt
            # draw datetime
            image, bottom_y = draw_background(image, f"{home_visitor[game.gameId][0]} vs {home_visitor[game.gameId][1]} - {str(game.gameId)}", (209, 222, 233), (242, 149, 89))
            draw_status(image, game, dt, home_visitor[game.gameId][0], bottom_y)
            cv2.putText(image, dt.strftime('%Y-%m-%d %H:%M:%S'), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (32, 44, 57), 2)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if not pipline.inited:
                height, width = image.shape[:2]
                pipline.build(width, height)
            repeat_times = round(ms_interval / one_frame_duration_ms)
            for _ in range(max(1, repeat_times)):
                pipline.write(image)

    def run_new_low_memory(self, gameid, tracking: TrackingNormData, pff: PffNormData, play: PlayNormData, game: GameNormData, player: PlayerNormData, merge: MergeNormData, targetX='jerseyNumber', targetY='pff_role', draw_once=False):
        df = merge.game.copy()
        df = df[df['gameId'] == gameid]
        home_visitor = merge.game_info[merge.game_info['gameId'] == gameid][['homeTeamAbbr', 'visitorTeamAbbr']].values[0]
        dts = df['time'].unique()
        start = dts[0]
        if self.time_max > 0:
            dts = [x for x in dts if x <= start + timedelta(seconds=self.time_max)]
        pipline = VideoPipline(self.encoder, self.output, self.fps)
        one_frame_duration_ms = 1000 / self.fps
        last_dt = dts[0]
        for dt in tqdm(dts):
            image = np.zeros((1106, 2400, 3), dtype=np.uint8)
            image.fill(255)
            draw_by_time_df(image, df, dt, home_visitor[0])
            ms_interval = (dt - last_dt).total_seconds() * 1000
            if ms_interval > 1000 * 5:
                ms_interval = 1000
            last_dt = dt
            # draw datetime
            image, bottom_y = draw_background(image, f"{home_visitor[0]} vs {home_visitor[1]} - {str(gameid)}", (209, 222, 233), (242, 149, 89))
            draw_status_df(image, df, dt, home_visitor[0], bottom_y, targetX=targetX, targetY=targetY, draw_once=draw_once)
            cv2.putText(image, dt.strftime('%Y-%m-%d %H:%M:%S'), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (32, 44, 57), 2)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if not pipline.inited:
                height, width = image.shape[:2]
                pipline.build(width, height)
            repeat_times = round(ms_interval / one_frame_duration_ms)
            for _ in range(max(1, repeat_times)):
                pipline.write(image)
            yield image.copy()

    # loading

    def load_demo_data_nfl_data(self) -> tuple[dict[int, GameNFLData], GameData]:
        weeks = [x for x in os.listdir(self.data_dir) if x.startswith('week')]
        if len(weeks) == 0:
            raise ValueError("No week file found")
        weeks = [os.path.join(self.data_dir, x) for x in weeks]
        pff_file = os.path.join(self.data_dir, 'pffScoutingData.csv')
        play_file = os.path.join(self.data_dir, 'plays.csv')
        game_file = os.path.join(self.data_dir, 'games.csv')
        gameNFLData = GameNFLData.loads(weeks, pff_file, play_file)
        game_data = GameData(game_file)
        return gameNFLData, game_data

    # tools

    def ffmpeg_encode(self, frames):
        height, width = frames[0][0].shape[:2]
        # FFmpeg命令，使用libx264编码器，并启用CUDA
        command = [
            'ffmpeg',
            '-y',  # 覆盖输出文件
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',  # 分辨率
            '-pix_fmt', 'bgr24',  # OpenCV的默认格式
            '-r', f'{self.fps}',  # 帧率
            '-i', '-',  # 输入来自stdin
            '-c:v', 'h264_nvenc',  # 使用NVIDIA的h264编码器
            '-pix_fmt', 'yuv420p',  # 输出像素格式
            f'{self.output}'  # 输出文件名
        ]
        one_frame_duration_ms = 1000 / self.fps
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        for frame, interval_ms in tqdm(frames):
            repeat_times = round(interval_ms / one_frame_duration_ms)
            for _ in range(max(1, repeat_times)):
                process.stdin.write(frame.tobytes())

        process.stdin.close()
        process.wait()

        if process.returncode != 0:
            raise RuntimeError('FFmpeg ERROR，Code {}'.format(process.returncode))

    def opencv_encode(self, frames):
        height, width = frames[0][0].shape[:2]
        one_frame_duration_ms = 1000 / self.fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'{self.output}', fourcc, self.fps, (width, height))
        for frame, interval_ms in tqdm(frames):
            repeat_times = round(interval_ms / one_frame_duration_ms)
            for _ in range(max(1, repeat_times)):
                out.write(frame)
        out.release()


def main(args):
    visual = Visual(args)
    if not args.lowmemory:
        visual.run()
    else:
        visual.run_low_memory()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize the game data")
    parser.add_argument("--data_dir", type=str, default="../data", help="The directory of the data")
    parser.add_argument("--output", type=str, default="output.mp4", help="The output file")
    parser.add_argument("--encoder", type=str, choices=["ffmpeg", "opencv"], default="ffmpeg", help="The encoder to use")
    parser.add_argument("--fps", type=int, default=25, help="The frames per second")
    parser.add_argument("--time_max", type=int, default=60 * 120, help="The max time to visualize")
    parser.add_argument("--lowmemory", action="store_true", help="Use low memory mode")
    parser.add_argument("--gameid", type=int)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
