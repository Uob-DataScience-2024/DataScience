import pandas as pd


class PffDataItem:
    pff_role: str
    pff_positionLinedUp: str
    pff_blockType: str

    pff_hit: bool
    pff_hurry: bool
    pff_sack: bool
    pff_beatenByDefender: bool
    pff_hitAllowed: bool
    pff_hurryAllowed: bool
    pff_sackAllowed: bool

    pff_nflIdBlockedPlayer: int

    no_payload_columns = {
        'gameId': 'gameId',
        'playId': 'playId',
        'nflId': 'nflId',
    }

    binary_list = [
        'pff_hit',
        'pff_hurry',
        'pff_sack',
        'pff_beatenByDefender',
        'pff_hitAllowed',
        'pff_hurryAllowed',
        'pff_sackAllowed',
    ]

    def __init__(self, gameId: int, playId: int, nflId: int, number_payload: dict, binary_payload: dict, text_payload: dict):
        self.gameId = gameId
        self.playId = playId
        self.nflId = nflId
        for key, value in number_payload.items():
            setattr(self, key, value)
        for key, value in text_payload.items():
            setattr(self, key, value)
        for key, value in binary_payload.items():
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

    def __str__(self):
        return f'PffDataItem(gameId={self.gameId}, playId={self.playId}, nflId={self.nflId}, pff_role={self.pff_role}, pff_positionLinedUp={self.pff_positionLinedUp}, pff_blockType={self.pff_blockType} | [{self.pff_hit}, {self.pff_hurry}, {self.pff_sack}, {self.pff_beatenByDefender}, {self.pff_hitAllowed}, {self.pff_hurryAllowed}, {self.pff_sackAllowed}, pff_nflIdBlockedPlayer={self.pff_nflIdBlockedPlayer}])'


class GamePffData:
    def __init__(self, gameId: int, df: pd.DataFrame):
        self.gameId = gameId
        self.df = df
        columns = df.columns
        headers = {}  # save the type of each column
        for column in columns:
            headers[column] = df[column].dtype
        self.number_list = list(filter((lambda x: pd.api.types.is_numeric_dtype(x[1]) and x[0] not in PffDataItem.no_payload_columns.values()), headers.items()))
        self.binary_category_list = list(filter((lambda x: pd.api.types.is_bool_dtype(x[1]) and x[0] not in PffDataItem.no_payload_columns.values()), headers.items()))
        self.text_list = list(filter((lambda x: not pd.api.types.is_numeric_dtype(x[1]) and x[0] not in PffDataItem.no_payload_columns.values()), headers.items()))
        self.no_payload_columns = list(PffDataItem.no_payload_columns.items())

    @staticmethod
    def load(filename):
        df = pd.read_csv(filename)
        loaded = {}
        for gameId in df['gameId'].unique():
            sub_df = df[df['gameId'] == gameId]
            loaded[gameId] = GamePffData(gameId, sub_df)
        return loaded

    def __str__(self):
        return f'GamePffData(gameId={self.gameId}, df=[len:{self.df.shape[0]}, number columns:{len(self.df.columns)}])'

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> PffDataItem:
        line = self.df.iloc[idx]
        args = {arg_name: line[col_name] for arg_name, col_name in self.no_payload_columns}
        args['number_payload'] = {col_name: line[col_name] for col_name, dtype in self.number_list}
        args['binary_payload'] = {col_name: line[col_name] for col_name, dtype in self.binary_category_list}
        args['text_payload'] = {col_name: line[col_name] for col_name, dtype in self.text_list}
        return PffDataItem(**args)