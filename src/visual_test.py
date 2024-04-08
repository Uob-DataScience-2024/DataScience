import os
import subprocess
import unittest
from datetime import timedelta

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from tqdm import tqdm

from visual.game_visualization import *
from data import GameNFLData, GameTrackingData, TrackingDataItem, GameFootBallTrackingData, GameData

data_dir = '../test_data'
if len(list(filter(lambda x: x.endswith('.csv'), os.listdir(data_dir)))) == 0:
    logger.warning('No csv file found in data directory, now run test data')
    data_dir = '../test_data'


def load_demo_data_nfl_data() -> tuple[dict[int, GameNFLData], GameData]:
    weeks = [x for x in os.listdir(data_dir) if x.startswith('week')]
    if len(weeks) == 0:
        raise ValueError("No week file found")
    weeks = [os.path.join(data_dir, x) for x in weeks]
    pff_file = os.path.join(data_dir, 'pffScoutingData.csv')
    play_file = os.path.join(data_dir, 'plays.csv')
    game_file = os.path.join(data_dir, 'games.csv')
    gameNFLData = GameNFLData.loads(weeks, pff_file, play_file)
    game_data = GameData(game_file)
    return gameNFLData, game_data


def ffmpeg_encode(frames: list[np.ndarray], fps = 25):
    height, width = frames[0][0].shape[:2]

    # FFmpeg命令，使用libx264编码器，并启用CUDA
    command = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',  # 分辨率
        '-pix_fmt', 'bgr24',  # OpenCV的默认格式
        '-r', f'{fps}',  # 帧率
        '-i', '-',  # 输入来自stdin
        '-c:v', 'h264_nvenc',  # 使用NVIDIA的h264编码器
        '-pix_fmt', 'yuv420p',  # 输出像素格式
        'output.mp4'  # 输出文件名
    ]
    one_frame_duration_ms = 1000 / fps
    # 启动子进程
    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    # 发送帧到FFmpeg
    for frame, interval_ms in tqdm(frames):
        # 计算需要重复该帧的次数
        repeat_times = round(interval_ms / one_frame_duration_ms)

        # 重复写入帧
        for _ in range(max(1, repeat_times)):
            process.stdin.write(frame.tobytes())

    # 关闭管道，完成编码
    process.stdin.close()
    process.wait()

    if process.returncode != 0:
        raise RuntimeError('FFmpeg错误，返回码 {}'.format(process.returncode))


class VisualTest(unittest.TestCase):
    # def test_first(self):
    #     config = TrainingConfigure.from_file('example.json')
    #     visualize_model('logdir/default_best(66.08%)_Seq2SeqGRU_2024-03-27_21-54-34.pt', config)
    #     self.assertEqual(True, True)  # add assertion here
    #
    # def test_visual_single_label(self):
    #     config = TrainingConfigure.from_file('example.json')
    #     visualize_model_single_label('logdir/default_best(66.08%)_Seq2SeqGRU_2024-03-27_21-54-34.pt', config)
    #     self.assertEqual(True, True)  # add assertion here

    def test(self):
        self.assertEqual(True, True)  # add assertion here

    def test_image(self):
        games, game_data = load_demo_data_nfl_data()
        game = games[list(games.keys())[0]]
        football = GameFootBallTrackingData.from_tracking_data(game.tracking)
        home_visitor = game_data.get_home_visitor()
        image = np.zeros((1106, 2400, 3), dtype=np.uint8)
        image.fill(255)
        # frameIds = game.df['frameId'].unique()
        # playIds = game.df['playId'].unique()
        # draw_frame(image, game, football, playIds[0], frameIds[0])
        dts = game.tracking.df['time'].unique()
        dt = dts[1]
        draw_by_time(image, game, football, dt, home_visitor[game.gameId][0])
        # dpi = 100
        image, bottom_y = draw_background(image, f"{home_visitor[game.gameId][0]} vs {home_visitor[game.gameId][1]} - {str(game.gameId)}", (209, 222, 233), (242, 149, 89))
        draw_status(image, game, dt, home_visitor[game.gameId][0], bottom_y)
        plt.figure(figsize=(24, 11.06))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        self.assertEqual(True, True)

    def test_video(self):
        games, game_data = load_demo_data_nfl_data()
        game = games[list(games.keys())[0]]
        football = GameFootBallTrackingData.from_tracking_data(game.tracking)
        home_visitor = game_data.get_home_visitor()

        # frameIds = game.df['frameId'].unique()
        # playIds = game.df['playId'].unique()
        # draw_frame(image, game, football, playIds[0], frameIds[0])
        dts = game.tracking.df['time'].unique()
        dts = list(sorted(dts))
        time_max = 60 * 70
        start = dts[0]
        dts = [x for x in dts if x <= start + timedelta(seconds=time_max)]
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
            result.append((image, ms_interval))
        # dpi = 100

        height, width = result[0][0].shape[:2]
        fps = 25
        one_frame_duration_ms = 1000 / fps
        # 创建视频写入对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

        for frame, interval_ms in tqdm(result):
            # 计算需要重复该帧的次数
            repeat_times = round(interval_ms / one_frame_duration_ms)

            # 重复写入帧
            for _ in range(max(1, repeat_times)):
                video.write(frame)

        video.release()
        self.assertEqual(True, True)

    def test_video_ffmpeg(self):
        games, game_data = load_demo_data_nfl_data()
        game = games[list(games.keys())[0]]
        football = GameFootBallTrackingData.from_tracking_data(game.tracking)
        home_visitor = game_data.get_home_visitor()

        # frameIds = game.df['frameId'].unique()
        # playIds = game.df['playId'].unique()
        # draw_frame(image, game, football, playIds[0], frameIds[0])
        dts = game.tracking.df['time'].unique()
        dts = list(sorted(dts))
        time_max = 60 * 120
        start = dts[2338]
        dts = [x for x in dts if start <= x <= start + timedelta(seconds=time_max)]
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
        # dpi = 100

        ffmpeg_encode(result)

        # height, width = result[0][0].shape[:2]
        # fps = 25
        # one_frame_duration_ms = 1000 / fps
        # # 创建视频写入对象
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
        #
        # for frame, interval_ms in tqdm(result):
        #     # 计算需要重复该帧的次数
        #     repeat_times = round(interval_ms / one_frame_duration_ms)
        #
        #     # 重复写入帧
        #     for _ in range(max(1, repeat_times)):
        #         video.write(frame)
        #
        # video.release()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
