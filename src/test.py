import os
import re
import unittest
import pandas as pd
from datastructure import TrackingDataItem, GameTrackingData
from tqdm import tqdm

from loguru import logger

data_dir = '../data'
if len(list(filter(lambda x: x.endswith('.csv'), os.listdir(data_dir)))) == 0:
    logger.warning('No csv file found in data directory, now run test data')
    data_dir = '../test_data'


class MainTest(unittest.TestCase):
    def test_TrackingDataItem(self):
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
        self.assertEqual(True, True)  # add assertion here

    def test_load_GameTrackingData(self):
        weeks = [x for x in os.listdir(data_dir) if x.startswith('week')]
        if len(weeks) == 0:
            self.fail("No week file found")
        test_week_file = weeks[0]
        test_week_file = os.path.join(data_dir, test_week_file)
        loaded = GameTrackingData.load(test_week_file)
        data = loaded[list(loaded.keys())[0]]
        item = data[0]
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
