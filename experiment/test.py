import os
import unittest

import torch
from loguru import logger

from dataset import DatasetPffBlockType
from model import Seq2Seq

data_dir = '../data'
if len(list(filter(lambda x: x.endswith('.csv'), os.listdir(data_dir)))) == 0:
    logger.warning('No csv file found in data directory, now run test data')
    data_dir = '../test_data'


class Experiment(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(f"data_dir: {data_dir}")
        self.dataset = DatasetPffBlockType(data_dir, cache=True)
        logger.info(f"Dataset loaded")

    def test_dataset(self):
        logger.info(f"len of dataset: {len(self.dataset)}")
        self.assertEqual(True, True)

    def test_statistics(self):
        # dataset = DatasetPffBlockType(data_dir)
        statistics = self.dataset.statistics()
        for key, value in statistics.items():
            logger.info(f'{key}: {value}')
        self.assertEqual(True, True)

    def test_category(self):
        # dataset = DatasetPffBlockType(data_dir)
        columns = self.dataset.get_column()
        for key, value in columns.items():
            if key == "union_id" or key == 'time':
                continue
            if value.name == 'string' or value.name == 'category' or value.name == 'object':
                category = self.dataset.get_category(key)
                logger.info(f"len of category[{key}]: {len(category)}")
                logger.info(f"category: {category}")
        self.assertEqual(True, True)


class ModelTest(unittest.TestCase):
    def test_model_init(self):
        model = Seq2Seq(10, 20, 30)
        tensor = torch.rand(10, 5, 10)
        output = model(tensor)
        logger.info(f"output shape: {output.shape}")
        self.assertEqual(output.shape, (10, 5, 30))


if __name__ == '__main__':
    unittest.main()
