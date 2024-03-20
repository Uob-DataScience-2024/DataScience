import os
import unittest

import torch
from loguru import logger

from dataset import DatasetPffBlockType, DatasetPffBlockTypeAutoSpilt
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
                category = self.dataset.get_categories(key)
                logger.info(f"len of category[{key}]: {len(category)}")
                logger.info(f"category: {category}")
        self.assertEqual(True, True)

    def test_category_table(self):
        # dataset = DatasetPffBlockType(data_dir)
        category_table = self.dataset.build_category_table()
        for key, value in category_table.items():
            logger.info(f"{key}: {value}")
        self.assertEqual(True, True)

    def test_all_item(self):
        # dataset = DatasetPffBlockType(data_dir)
        for item in self.dataset:
            logger.info(f"item: {item[0].shape, item[1].shape}")
        self.assertEqual(True, True)


class NewDatasetTest(unittest.TestCase):
    def test_data(self):
        dataset = DatasetPffBlockTypeAutoSpilt("../test_data", cache=False)
        d11 = dataset[0]
        d12 = dataset[1]
        d13 = dataset[2]
        d14 = dataset[3]
        d21 = dataset[4]
        d22 = dataset[5]
        d23 = dataset[6]
        d24 = dataset[7]
        logger.info(f"d11: {d11[0].shape, d11[1].shape}")
        logger.info(f"d12: {d12[0].shape, d12[1].shape}")
        logger.info(f"d13: {d13[0].shape, d13[1].shape}")
        logger.info(f"d14: {d14[0].shape, d14[1].shape}")
        logger.info(f"d21: {d21[0].shape, d21[1].shape}")
        logger.info(f"d22: {d22[0].shape, d22[1].shape}")
        logger.info(f"d23: {d23[0].shape, d23[1].shape}")
        logger.info(f"d24: {d24[0].shape, d24[1].shape}")
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
