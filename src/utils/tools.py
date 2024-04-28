import os
from data.total_data import SplitMode, TrackingNormData, PffNormData, PlayNormData, PlayerNormData, GameNormData, MergeNormData


def load_data(data_dir: str, week_limit=0):
    weeks = [x for x in os.listdir(data_dir) if x.startswith('week')]
    if len(weeks) == 0:
        raise ValueError("No week file found")
    weeks = [os.path.join(data_dir, x) for x in weeks]
    if week_limit > 0:
        weeks = weeks[:week_limit]
    pff_file = os.path.join(data_dir, 'pffScoutingData.csv')
    play_file = os.path.join(data_dir, 'plays.csv')
    game_file = os.path.join(data_dir, 'games.csv')
    player_file = os.path.join(data_dir, 'players.csv')
    tracking = TrackingNormData(weeks, parallel_loading=True)
    pff = PffNormData(pff_file)
    play = PlayNormData(play_file)
    game = GameNormData(game_file)
    player = PlayerNormData(player_file)
    merge = MergeNormData(player, game, tracking, pff, play)
    return tracking, pff, play, game, player, merge
