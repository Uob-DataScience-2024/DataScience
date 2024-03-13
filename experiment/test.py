import unittest
from loguru import logger

from dataset import DatasetPffBlockType

data_dir = '../data'


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


if __name__ == '__main__':
    unittest.main()
