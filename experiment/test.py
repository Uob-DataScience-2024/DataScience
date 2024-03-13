import os
import unittest
from loguru import logger

from dataset import DatasetPffBlockType

data_dir = '../data'
if len(list(filter(lambda x: x.endswith('.csv'), os.listdir(data_dir)))) == 0:
    logger.warning('No csv file found in data directory, now run test data')
    data_dir = '../test_data'


class Experiment(unittest.TestCase):
    def test_dataset(self):
        dataset = DatasetPffBlockType(data_dir)
        self.assertEqual(True, True)

    def test_statistics(self):
        dataset = DatasetPffBlockType(data_dir)
        statistics = dataset.statistics()
        for key, value in statistics.items():
            logger.info(f'{key}: {value}')
        self.assertEqual(True, True)

    def test_category(self):
        dataset = DatasetPffBlockType(data_dir)
        columns = dataset.get_column()
        for key, value in columns.items():
            if key == "union_id" or key == 'time':
                continue
            if value.name == 'string' or value.name == 'category' or value.name == 'object':
                category = dataset.get_category(key)
                logger.info(f"len of category[{key}]: {len(category)}")
                logger.info(f"category: {category}")
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
