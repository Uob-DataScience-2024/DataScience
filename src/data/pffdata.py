import numpy as np
import pandas as pd
import torch
from loguru import logger


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

    resize_range = {

    }

    category_labels = {
    }

    block_columns = ['gameId']

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

    def statistics(self):
        return self.df.describe()

    def tensor(self, resize_range_overwrite: dict, category_labels_overwrite: dict, columns: list[str] = None, dtype=torch.float32) -> [torch.Tensor, dict[str, dict]]:
        df = self.df.copy()
        if columns is None:
            columns = df.columns
        elif len(list(filter(lambda x: x not in df.columns, columns))):
            raise ValueError(f"Columns not found in dataframe: {list(filter(lambda x: x not in df.columns, columns))}")
        columns = list(filter(lambda x: x not in PffDataItem.block_columns, columns))
        types = {column: df[column].dtype for column in columns}
        statistics = self.statistics()
        resize_range = PffDataItem.resize_range.copy()
        resize_range.update(resize_range_overwrite)
        category_labels: dict[str, None | list] = PffDataItem.category_labels.copy()
        category_labels.update(category_labels_overwrite)
        data_map = {}
        for i, column in enumerate(columns):
            if pd.api.types.is_numeric_dtype(types[column]):
                if column in resize_range.keys() and resize_range[column] is not None:
                    vMin, vMax = resize_range[column]
                    df[column] = (df[column] - vMin) / (vMax - vMin)
                else:
                    vMin, vMax = statistics[column]['min'], statistics[column]['max']
                    df[column] = (df[column] - vMin) / (vMax - vMin)
                data_map[column] = {
                    "type": "number",
                    "min": vMin,
                    "max": vMax,
                    "index": i,
                }
            elif pd.api.types.is_string_dtype(types[column]):
                if column in category_labels.keys() and category_labels[column] is not None:
                    label_mapping = {k: v for v, k in enumerate(category_labels[column])}
                    df[column] = df[column].map(label_mapping)
                    data_map[column] = {
                        "type": "category",
                        "labels": category_labels[column],
                        "index": i,
                    }
                else:
                    labels = df[column].unique()
                    if pd.isna(labels).any():
                        labels = labels[1:].tolist()
                    category = pd.Categorical(df[column], categories=labels)
                    df[column] = category.codes
                    data_map[column] = {
                        "type": "category",
                        "labels": labels,
                        "index": i,
                    }
            elif pd.api.types.is_bool_dtype(types[column]):
                df[column] = df[column].astype(np.int8)
                data_map[column] = {
                    "type": "binary",
                    "index": i,
                }
            elif pd.api.types.is_datetime64_any_dtype(types[column]):
                # to timestamp ms
                df[column] = df[column].astype(np.int64) // 10 ** 6
                start = df[column].min()
                end = df[column].max()
                df[column] = ((df[column] - start) / (end - start)) * 10
                data_map[column] = {
                    "type": "datetime",
                    "mode": "timestamp",
                    "index": i,
                }
            else:
                logger.warning(f"Column {column} is not a valid type")
            if not pd.api.types.is_numeric_dtype(df[column].dtype):
                pass
        # all nan to -1
        df = df.fillna(-1)
        df = df[columns]
        tensor = torch.tensor(df.values, dtype=dtype)
        return tensor, data_map
