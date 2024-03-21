import os
import re
import unittest
import pandas as pd

from datastructure import NFLDataItem, GameNFLData
from playdata import PlayDataItem, GamePlayData
from pffdata import PffDataItem, GamePffData
from trackingdata import TrackingDataItem, GameTrackingData
from tqdm import tqdm

from loguru import logger

data_dir = '../data'
if len(list(filter(lambda x: x.endswith('.csv'), os.listdir(data_dir)))) == 0:
    logger.warning('No csv file found in data directory, now run test data')
    data_dir = '../test_data'


class MainTest(unittest.TestCase):
    def test_TrackingDataItem(self):
        logger.info('Testing TrackingDataItem')
        weeks = [x for x in os.listdir(data_dir) if x.startswith('week')]
        if len(weeks) == 0:
            self.fail("No week file found")
        test_week_file = weeks[0]
        test_week_file = os.path.join(data_dir, test_week_file)
        week = re.search(r'week(\d+)', test_week_file).group(1)
        df = pd.read_csv(test_week_file)
        # set time to datetime
        df['time'] = pd.to_datetime(df['time'])
        columns = df.columns
        headers = {}  # save the type of each column
        for column in columns:
            headers[column] = df[column].dtype
        number_list = list(filter((lambda x: pd.api.types.is_numeric_dtype(x[1]) and x[0] not in TrackingDataItem.no_payload_columns.values()), headers.items()))
        text_list = list(filter((lambda x: not pd.api.types.is_numeric_dtype(x[1]) and x[0] not in TrackingDataItem.no_payload_columns.values()), headers.items()))
        no_payload_columns = list(TrackingDataItem.no_payload_columns.items())
        items = []
        max_out = 10 ** 4
        for i, line in tqdm(df.iterrows(), total=max_out):
            args = {arg_name: line[col_name] for arg_name, col_name in no_payload_columns}
            args['number_payload'] = {col_name: line[col_name] for col_name, dtype in number_list}
            args['text_payload'] = {col_name: line[col_name] for col_name, dtype in text_list}
            item = TrackingDataItem(week, **args)
            items.append(item)
            if i >= max_out:
                break
        logger.info(f"Loaded: {items[0]}")
        logger.info('Testing TrackingDataItem done')
        self.assertEqual(True, True)

    def test_load_GameTrackingData(self):
        logger.info('Testing GameTrackingData')
        weeks = [x for x in os.listdir(data_dir) if x.startswith('week')]
        if len(weeks) == 0:
            self.fail("No week file found")
        test_week_file = weeks[0]
        test_week_file = os.path.join(data_dir, test_week_file)
        loaded = GameTrackingData.load(test_week_file)
        data = loaded[list(loaded.keys())[0]]
        item = data[0]
        logger.info('Testing GameTrackingData done')
        self.assertEqual(True, True)

    def test_PffDataItem(self):
        logger.info('Testing PffDataItem')
        filename = os.path.join(data_dir, 'pffScoutingData.csv')
        df = pd.read_csv(filename)
        columns = df.columns
        headers = {}  # save the type of each column
        for column in columns:
            headers[column] = df[column].dtype
        number_list = list(filter((lambda x: pd.api.types.is_numeric_dtype(x[1]) and x[0] not in PffDataItem.no_payload_columns.values()), headers.items()))
        binary_category_list = list(filter((lambda x: pd.api.types.is_bool_dtype(x[1]) and x[0] not in PffDataItem.no_payload_columns.values()), headers.items()))
        text_list = list(filter((lambda x: not pd.api.types.is_numeric_dtype(x[1]) and x[0] not in PffDataItem.no_payload_columns.values()), headers.items()))
        no_payload_columns = list(PffDataItem.no_payload_columns.items())
        items = []
        max_out = 10 ** 4
        for i, line in tqdm(df.iterrows(), total=max_out):
            args = {arg_name: line[col_name] for arg_name, col_name in no_payload_columns}
            args['number_payload'] = {col_name: line[col_name] for col_name, dtype in number_list}
            args['binary_payload'] = {col_name: line[col_name] for col_name, dtype in binary_category_list}
            args['text_payload'] = {col_name: line[col_name] for col_name, dtype in text_list}
            item = PffDataItem(**args)
            items.append(item)
            if i >= max_out:
                break
        logger.info('Testing PffDataItem done')
        self.assertEqual(True, True)

    def test_load_GamePffData(self):
        logger.info('Testing GamePffData')
        filename = os.path.join(data_dir, 'pffScoutingData.csv')
        loaded = GamePffData.load(filename)
        data = loaded[list(loaded.keys())[0]]
        item = data[0]
        logger.info('Testing GamePffData done')
        self.assertEqual(True, True)

    def test_PlayDataItem(self):
        logger.info('Testing PlayDataItem')
        filename = os.path.join(data_dir, 'plays.csv')
        df = pd.read_csv(filename)
        columns = df.columns
        headers = {}
        for column in columns:
            headers[column] = df[column].dtype
        number_list = list(filter((lambda x: pd.api.types.is_numeric_dtype(x[1]) and x[0] not in PlayDataItem.no_payload_columns.values()), headers.items()))
        binary_category_list = list(filter((lambda x: pd.api.types.is_bool_dtype(x[1]) and x[0] not in PlayDataItem.no_payload_columns.values()), headers.items()))
        text_list = list(filter((lambda x: not pd.api.types.is_numeric_dtype(x[1]) and x[0] not in PlayDataItem.no_payload_columns.values()), headers.items()))
        no_payload_columns = list(PlayDataItem.no_payload_columns.items())
        items = []
        max_out = 10 ** 4
        for i, line in tqdm(df.iterrows(), total=max_out):
            args = {arg_name: line[col_name] for arg_name, col_name in no_payload_columns}
            args['number_payload'] = {col_name: line[col_name] for col_name, dtype in number_list}
            args['binary_payload'] = {col_name: line[col_name] for col_name, dtype in binary_category_list}
            args['text_payload'] = {col_name: line[col_name] for col_name, dtype in text_list}
            item = PlayDataItem(**args)
            items.append(item)
            if i >= max_out:
                break
        logger.info('Testing PlayDataItem done')
        self.assertEqual(True, True)

    def test_load_GamePlayData(self):
        logger.info('Testing GamePlayData')
        filename = os.path.join(data_dir, 'plays.csv')
        loaded = GamePlayData.load(filename)
        data = loaded[list(loaded.keys())[0]]
        item = data[0]
        logger.info('Testing GamePlayData done')
        self.assertEqual(True, True)

    def test_NFLDataItem(self):
        logger.info('Testing NFLDataItem')
        weeks = [x for x in os.listdir(data_dir) if x.startswith('week')]
        if len(weeks) == 0:
            self.fail("No week file found")
        test_week_file = weeks[0]
        test_week_file = os.path.join(data_dir, test_week_file)
        tracking = list(GameTrackingData.load(test_week_file).values())[0]
        pff = list(GamePffData.load(os.path.join(data_dir, 'pffScoutingData.csv')).values())[0]
        play = list(GamePlayData.load(os.path.join(data_dir, 'plays.csv')).values())[0]
        nfl_item = NFLDataItem.from_object(tracking[0], pff[0], play[0])
        logger.info('Testing NFLDataItem done')
        self.assertEqual(True, True)

    def test_load_GameNFLData(self):
        logger.info('Testing GameNFLData')
        weeks = [x for x in os.listdir(data_dir) if x.startswith('week')]
        if len(weeks) == 0:
            self.fail("No week file found")
        test_week_file = weeks[0]
        test_week_file = os.path.join(data_dir, test_week_file)
        pff_file = os.path.join(data_dir, 'pffScoutingData.csv')
        play_file = os.path.join(data_dir, 'plays.csv')
        gameNFLData = GameNFLData.load(test_week_file, pff_file, play_file)
        data = gameNFLData[list(gameNFLData.keys())[0]]
        item = data[0]
        logger.info('Testing GameNFLData done')
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
