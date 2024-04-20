import pandas as pd


class PlayerData:
    def __init__(self, player_data_path):
        self.player_data_path = player_data_path
        self.df = pd.read_csv(player_data_path)


