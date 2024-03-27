import pandas as pd


class GameData:
    def __init__(self, game_data_path):
        self.game_data_path = game_data_path
        self.df = pd.read_csv(game_data_path)

    def get_home_visitor(self):
        df = self.df.copy()
        df = df[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']]
        result = {}
        for index, row in df.iterrows():
            result[row['gameId']] = (row['homeTeamAbbr'], row['visitorTeamAbbr'])
        return result
