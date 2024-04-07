import os
import unittest
from datetime import timedelta

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import TrainingConfigure
from visual import visualize_model, visualize_model_single_label
from visual.game_visualization import *
from data import GameNFLData, GameTrackingData, TrackingDataItem, GameFootBallTrackingData

data_dir = '../data'
if len(list(filter(lambda x: x.endswith('.csv'), os.listdir(data_dir)))) == 0:
    logger.warning('No csv file found in data directory, now run test data')
    data_dir = '../test_data'


def load_demo_data_nfl_data() -> dict[int, GameNFLData]:
    weeks = [x for x in os.listdir(data_dir) if x.startswith('week')]
    if len(weeks) == 0:
        raise ValueError("No week file found")
    weeks = [os.path.join(data_dir, x) for x in weeks]
    pff_file = os.path.join(data_dir, 'pffScoutingData.csv')
    play_file = os.path.join(data_dir, 'plays.csv')
    gameNFLData = GameNFLData.loads(weeks, pff_file, play_file)
    return gameNFLData


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

    # def test_image(self):
    #     games = load_demo_data_nfl_data()
    #     game = games[list(games.keys())[0]]
    #     football = GameFootBallTrackingData.from_tracking_data(game.tracking)
    #     image = np.zeros((1106, 2400, 3), dtype=np.uint8)
    #     image.fill(255)
    #     # frameIds = game.df['frameId'].unique()
    #     # playIds = game.df['playId'].unique()
    #     # draw_frame(image, game, football, playIds[0], frameIds[0])
    #     dts = game.tracking.df['time'].unique()
    #     draw_by_time(image, game, football, dts[1])
    #     # dpi = 100

    #     plt.figure(figsize=(24, 11.06))
    #     plt.imshow(image)
    #     plt.axis('off')
    #     plt.show()
    #     self.assertEqual(True, True)

    def test_video(self):
        games = load_demo_data_nfl_data()
        game = games[list(games.keys())[0]]
        football = GameFootBallTrackingData.from_tracking_data(game.tracking)

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
            draw_by_time(image, game, football, dt)
            ms_interval = (dt - last_dt).total_seconds() * 1000
            if ms_interval > 1000 * 5:
                ms_interval = 1000
            last_dt = dt
            # draw datetime
            cv2.putText(image, dt.strftime('%Y-%m-%d %H:%M:%S'), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
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

        # plt.figure(figsize=(24, 11.06))
        # plt.imshow(image)
        # plt.axis('off')
        # plt.show()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
