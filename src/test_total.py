import os
import time
import unittest
from data.total_data import SplitMode, TrackingNormData, PffNormData, PlayNormData, PlayerNormData, GameNormData, MergeNormData

from loguru import logger

data_dir = '../data'
if len(list(filter(lambda x: x.endswith('.csv'), os.listdir(data_dir)))) == 0:
    logger.warning('No csv file found in data directory, now run test data')
    data_dir = '../test_data'

def get_files():
    weeks = [x for x in os.listdir(data_dir) if x.startswith('week')]
    if len(weeks) == 0:
        raise ValueError("No week file found")
    weeks = [os.path.join(data_dir, x) for x in weeks]
    pff_file = os.path.join(data_dir, 'pffScoutingData.csv')
    play_file = os.path.join(data_dir, 'plays.csv')
    game_file = os.path.join(data_dir, 'games.csv')
    player_file = os.path.join(data_dir, 'players.csv')
    return weeks, pff_file, play_file, game_file, player_file

class TestTotal(unittest.TestCase):
    def test_something(self):
        weeks, pff_file, play_file, game_file, player_file = get_files()
        start = time.time()
        tracking = TrackingNormData(weeks, parallel_loading=True)
        logger.info(f"tracking time: {time.time() - start} seconds")
        pff = PffNormData(pff_file)
        play = PlayNormData(play_file)
        game = GameNormData(game_file)
        player = PlayerNormData(player_file)
        start_merge = time.time()
        merge = MergeNormData(player, game, tracking, pff, play)
        end = time.time()
        logger.info(f"Merge time: {end - start_merge} seconds")
        logger.info(f"Total time: {end - start} seconds")
        self.assertEqual(True, True)  # add assertion here

    def test_split(self):
        weeks, pff_file, play_file, game_file, player_file = get_files()
        tracking = TrackingNormData(weeks)
        pff = PffNormData(pff_file)
        play = PlayNormData(play_file)
        game = GameNormData(game_file)
        player = PlayerNormData(player_file)
        merge = MergeNormData(player, game, tracking, pff, play)

        # spilt tracking data
        tracking_split_game = tracking.split(SplitMode.Game)
        tracking_split_play = tracking.split(SplitMode.PlayID)
        tracking_split_time = tracking.split(SplitMode.Time)
        tracking_split_time_stage = tracking.split(SplitMode.TimeStage)
        tracking_split_nflid = tracking.split(SplitMode.NflID)
        tracking_split_frameid = tracking.split(SplitMode.FrameID)
        self.assertEqual(True, True)

    def test_linemen_on_pass_plays(self):
        pass


if __name__ == '__main__':
    unittest.main()
