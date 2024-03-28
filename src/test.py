import os
import re
import unittest
import pandas as pd
import torch

from data import NFLDataItem, GameNFLData, PlayDataItem, GamePlayData, PffDataItem, GamePffData, TrackingDataItem, GameTrackingData
from tqdm import tqdm

from network import Seq2SeqLSTM
from utils.training_config import TrainingConfigure

from loguru import logger

data_dir = '../data'
if len(list(filter(lambda x: x.endswith('.csv'), os.listdir(data_dir)))) == 0:
    logger.warning('No csv file found in data directory, now run test data')
    data_dir = '../test_data'


class DataClassInitTest(unittest.TestCase):
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
        logger.info(f'GameTrackingData: {data}')
        logger.info(f'First Item: {item}')
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
        logger.info(f"Loaded: {items[0]}")
        logger.info('Testing PffDataItem done')
        self.assertEqual(True, True)

    def test_load_GamePffData(self):
        logger.info('Testing GamePffData')
        filename = os.path.join(data_dir, 'pffScoutingData.csv')
        loaded = GamePffData.load(filename)
        data = loaded[list(loaded.keys())[0]]
        item = data[0]
        logger.info(f'GamePffData: {data}')
        logger.info(f'First Item: {item}')
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
        logger.info(f"Loaded: {items[0]}")
        logger.info('Testing PlayDataItem done')
        self.assertEqual(True, True)

    def test_load_GamePlayData(self):
        logger.info('Testing GamePlayData')
        filename = os.path.join(data_dir, 'plays.csv')
        loaded = GamePlayData.load(filename)
        data = loaded[list(loaded.keys())[0]]
        item = data[0]
        logger.info(f'GamePlayData: {data}')
        logger.info(f'First Item: {item}')
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
        logger.info(f"NFLDataItem: {nfl_item}")
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
        logger.info(f'GameNFLData: {data}')
        logger.info(f'First Item: {item}')
        logger.info('Testing GameNFLData done')
        self.assertEqual(True, True)

    def test_load_all(self):
        logger.info('Testing load all')
        weeks = [x for x in os.listdir(data_dir) if x.startswith('week')]
        if len(weeks) == 0:
            self.fail("No week file found")
        weeks = [os.path.join(data_dir, x) for x in weeks]
        pff_file = os.path.join(data_dir, 'pffScoutingData.csv')
        play_file = os.path.join(data_dir, 'plays.csv')
        gameNFLData = GameNFLData.loads(weeks, pff_file, play_file)
        data = gameNFLData[list(gameNFLData.keys())[0]]
        item = data[0]
        logger.info(f'GameNFLData: {data}')
        logger.info(f'First Item: {item}')
        logger.info('Testing load all done')
        self.assertEqual(True, True)


def load_demo_data_tracking_data() -> dict[int, GameTrackingData]:
    weeks = [x for x in os.listdir(data_dir) if x.startswith('week')]
    if len(weeks) == 0:
        raise ValueError("No week file found")
    test_week_file = weeks[0]
    test_week_file = os.path.join(data_dir, test_week_file)
    loaded = GameTrackingData.load(test_week_file)
    return loaded


def load_demo_data_pff_data() -> dict[int, GamePffData]:
    filename = os.path.join(data_dir, 'pffScoutingData.csv')
    loaded = GamePffData.load(filename)
    return loaded


def load_demo_data_play_data() -> dict[int, GamePlayData]:
    filename = os.path.join(data_dir, 'plays.csv')
    loaded = GamePlayData.load(filename)
    return loaded


def load_demo_data_nfl_data() -> dict[int, GameNFLData]:
    weeks = [x for x in os.listdir(data_dir) if x.startswith('week')]
    if len(weeks) == 0:
        raise ValueError("No week file found")
    weeks = [os.path.join(data_dir, x) for x in weeks]
    pff_file = os.path.join(data_dir, 'pffScoutingData.csv')
    play_file = os.path.join(data_dir, 'plays.csv')
    gameNFLData = GameNFLData.loads(weeks, pff_file, play_file)
    return gameNFLData


class DataClassMethodTest(unittest.TestCase):
    def test_statistics(self):
        data = load_demo_data_tracking_data()
        demo: GameTrackingData = data[list(data.keys())[0]]
        statistics = demo.statistics()
        logger.info(f"Statistics: {statistics}")
        self.assertEqual(True, True)

    def test_tensor_tracking_data(self):
        data = load_demo_data_tracking_data()
        demo: GameTrackingData = data[list(data.keys())[0]]
        tensor, data_map = demo.tensor({}, {})
        logger.info(f"Tensor: {tensor.shape}")
        logger.info(f"Data Map: {data_map}")
        self.assertEqual(True, True)

    def test_tensor_pff_data(self):
        data = load_demo_data_pff_data()
        demo: GamePffData = data[list(data.keys())[0]]
        tensor, data_map = demo.tensor({}, {})
        logger.info(f"Tensor: {tensor.shape}")
        logger.info(f"Data Map: {data_map}")
        self.assertEqual(True, True)

    def test_tensor_play_data(self):
        data = load_demo_data_play_data()
        demo: GamePlayData = data[list(data.keys())[0]]
        tensor, data_map = demo.tensor({}, {})
        logger.info(f"Tensor: {tensor.shape}")
        logger.info(f"Data Map: {data_map}")
        self.assertEqual(True, True)

    def test_tensor_nfl_data(self):
        data = load_demo_data_nfl_data()
        demo: GameNFLData = data[list(data.keys())[0]]
        tensor, data_map = demo.tensor({}, {})
        logger.info(f"Tensor: {tensor.shape}")
        logger.info(f"Data Map: {data_map}")
        self.assertEqual(True, True)

    def test_get_quarter_partition(self):
        data = load_demo_data_play_data()
        demo: GamePlayData = data[list(data.keys())[0]]
        partition = demo.get_quarter_partition()
        logger.info(f"Quarter Partition: {partition}")
        self.assertEqual(True, True)

    def test_tracking_filter(self):
        tracking_data = load_demo_data_tracking_data()
        playdata = load_demo_data_play_data()
        game_id = list(tracking_data.keys())[0]
        partition = playdata[game_id].get_quarter_partition()
        tensor = tracking_data[game_id].tensor({}, {})
        tensor_filtered = tracking_data[game_id].tensor({}, {}, play_id_filter=partition[0])
        logger.info(f"partition range: {[x.shape for x in partition]}")
        logger.info(f"Tensors: {tensor[0].shape} -> {tensor_filtered[0].shape}")
        self.assertEqual(True, True)

    def test_mask(self):
        nfl_data = load_demo_data_nfl_data()
        game_id = list(nfl_data.keys())[0]
        mask = nfl_data[game_id].union_id_mask()
        logger.info(f"Mask: {len(mask)}")
        self.assertEqual(True, True)


class TestDataset(unittest.TestCase):
    def test_TrackingDataset(self):
        from network import TrackingDataset
        dataset = TrackingDataset(data_dir)
        x, y = dataset[0]
        logger.info(f"X: {x.shape}, Y: {y.shape}")
        self.assertEqual(True, True)

    # def test_sequence_dataset(self):
    #     from dataset import SequenceDataset
    #     dataset = SequenceDataset(data_dir)
    #     x, y = dataset[0]
    #     logger.info(f"X: {x.shape}, Y: {y.shape}")
    #     self.assertEqual(True, True)

    # def test_sequence_dataset_split(self):
    #     from dataset import SequenceDataset
    #     dataset_split = SequenceDataset(data_dir, split=True)
    #     dataset_no_split = SequenceDataset(data_dir, split=False)
    #     logger.info(f"Split Dataset: {len(dataset_split)}, No Split Dataset: {len(dataset_no_split)}")
    #     x, y = dataset_no_split[0]
    #     logger.info(f"X: {x.shape}, Y: {y.shape}")
    #     X1, Y1 = dataset_split[0]
    #     X2, Y2 = dataset_split[1]
    #     X3, Y3 = dataset_split[2]
    #     X4, Y4 = dataset_split[3]
    #     logger.info(f"X1: {X1.shape}, Y1: {Y1.shape}")
    #     logger.info(f"X2: {X2.shape}, Y2: {Y2.shape}")
    #     logger.info(f"X3: {X3.shape}, Y3: {Y3.shape}")
    #     logger.info(f"X4: {X4.shape}, Y4: {Y4.shape}")
    #     logger.info(f"Sum: {X1.shape[0] + X2.shape[0] + X3.shape[0] + X4.shape[0]}, No Split: {x.shape[0]}")
    #     self.assertEqual(X1.shape[0] + X2.shape[0] + X3.shape[0] + X4.shape[0], x.shape[0])

    def test_sequence_dataset_custom_col(self):
        from network import SequenceDataset
        input_features = ['playId', 'nflId', 'frameId', 'time', 'jerseyNumber', 'team', 'playDirection', 'x', 'y', 's', 'a', 'dis', 'o', 'dir']
        logger.info(f"Input Features({len(input_features)}): {input_features}")
        dataset = SequenceDataset(data_dir, input_features=input_features, target_feature='pff_blockType')
        x, y = dataset[0]
        logger.info(f"X: {x.shape}, Y: {y.shape}")
        self.assertEqual(True, True)

    def test_segment_dataset(self):
        from network import SegmentDataset
        input_features = ['playId', 'nflId', 'frameId', 'time', 'jerseyNumber', 'team', 'playDirection', 'x', 'y', 's', 'a', 'dis', 'o', 'dir']
        dataset = SegmentDataset(data_dir, input_features=input_features, target_feature='pff_blockType')
        x, y = dataset[832]
        logger.info(f"X: {x.shape}, Y: {y.shape}, Item max length: {dataset.item_max_len}")
        self.assertEqual(True, True)


class TestNetwork(unittest.TestCase):
    def test_same_size_cnn(self):
        from network import SameSizeCNN
        model = SameSizeCNN(1, 32, 12)
        data = torch.randn(1, 1, 203, 14)
        out = model(data)
        logger.info(f"Out: {out.shape}")
        self.assertEqual(True, True)

    def test_config(self):
        config = TrainingConfigure()
        logger.info(f"Config: {config.to_json()}")
        self.assertEqual(True, True)

    def test_captum(self):
        from captum.attr import IntegratedGradients
        model = Seq2SeqLSTM(14, 32, 14, num_layers=2)
        input_data = torch.randn(1, 203, 14)

        ig = IntegratedGradients(model)
        output = model(input_data)
        attributions, delta = ig.attribute(input_data, target=(0, 0), return_convergence_delta=True)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
