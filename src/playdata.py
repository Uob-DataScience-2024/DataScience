import pandas as pd


class PlayDataItem:
    quarter: int
    down: int
    yardsToGo: int
    yardlineNumber: int
    preSnapHomeScore: int
    preSnapVisitorScore: int
    penaltyYards: int
    prePenaltyPlayResult: int
    playResult: int
    foulNFLId1: int
    foulNFLId2: int
    foulNFLId3: int
    absoluteYardlineNumber: int
    defendersInTheBox: int

    pff_playAction: bool

    playDescription: str
    possessionTeam: str
    defensiveTeam: str
    yardlineSide: str
    passResult: str
    foulName1: str
    foulName2: str
    foulName3: str
    offenseFormation: str
    personnelO: str
    personnelD: str
    pff_passCoverage: str
    pff_passCoverageType: str

    no_payload_columns = {
        'gameId': 'gameId',
        'playId': 'playId',
    }

    binary_list = [
        'pff_playAction',
    ]

    def __init__(self, gameId: int, playId: int, number_payload: dict, binary_payload: dict, text_payload: dict):
        self.gameId = gameId
        self.playId = playId
        for key, value in number_payload.items():
            setattr(self, key, value)
        for key, value in binary_payload.items():
            setattr(self, key, value)
        for key, value in text_payload.items():
            setattr(self, key, value)
        self.number_payload = number_payload
        self.binary_payload = binary_payload
        self.text_payload = text_payload
        for key in self.binary_list:
            if type(getattr(self, key)) == float:
                if getattr(self, key) == 1.0:
                    setattr(self, key, True)
                elif getattr(self, key) == 0.0:
                    setattr(self, key, False)


class GamePlayData:
    def __init__(self, gameId: int, df: pd.DataFrame):
        self.gameId = gameId
        self.df = df
        columns = df.columns
        headers = {}
        for column in columns:
            headers[column] = df[column].dtype
        self.number_list = list(filter((lambda x: pd.api.types.is_numeric_dtype(x[1]) and x[0] not in PlayDataItem.no_payload_columns.values()), headers.items()))
        self.binary_category_list = list(filter((lambda x: pd.api.types.is_bool_dtype(x[1]) and x[0] not in PlayDataItem.no_payload_columns.values()), headers.items()))
        self.text_list = list(filter((lambda x: not pd.api.types.is_numeric_dtype(x[1]) and x[0] not in PlayDataItem.no_payload_columns.values()), headers.items()))
        self.no_payload_columns = list(PlayDataItem.no_payload_columns.items())

    @staticmethod
    def load(filename):
        df = pd.read_csv(filename)
        loaded = {}
        for gameId in df['gameId'].unique():
            sub_df = df[df['gameId'] == gameId]
            loaded[gameId] = GamePlayData(gameId, sub_df)
        return loaded

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> PlayDataItem:
        line = self.df.iloc[idx]
        args = {arg_name: line[col_name] for arg_name, col_name in self.no_payload_columns}
        args['number_payload'] = {col_name: line[col_name] for col_name, dtype in self.number_list}
        args['binary_payload'] = {col_name: line[col_name] for col_name, dtype in self.binary_category_list}
        args['text_payload'] = {col_name: line[col_name] for col_name, dtype in self.text_list}
        return PlayDataItem(**args)
