import unittest

from dataset import DatasetPffBlockType

data_dir = '../data'


class Experiment(unittest.TestCase):
    def test_dataset(self):
        dataset = DatasetPffBlockType(data_dir)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
