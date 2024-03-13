import os.path
from datetime import datetime

from torch.utils.data import Dataset
import pandas as pd


class DatasetPffBlockType(Dataset):
    def __init__(self, data_dir, pff_filename='pffScoutingData.csv', tracking_filename='week{}.csv', cache=False):
        self.data_dir = data_dir
        self.pff_filename = pff_filename
        self.tracking_filename = tracking_filename
        self.final_data = {}
        self.raw_data = {
            'pff': self.load_pff_data(),
            'tracking': self.load_tracking_data()
        }
        self.postprocess(cache)

    def load_pff_data(self):
        return pd.read_csv(os.path.join(self.data_dir, self.pff_filename))

    def load_tracking_data(self, max_week=8):
        datas = []
        for i in range(1, max_week + 1):
            file = os.path.join(self.data_dir, self.tracking_filename.format(i))
            if os.path.exists(file) and os.path.isfile(file):
                datas.append(pd.read_csv(os.path.join(self.data_dir, self.tracking_filename.format(i))))
        # concat all the dataframes
        return pd.concat(datas, ignore_index=True)

    def load_cache(self):
        if not os.path.exists(os.path.join(self.data_dir, 'cache') or not os.path.isdir(os.path.join(self.data_dir, 'cache'))):
            os.mkdir(os.path.join(self.data_dir, 'cache'))
        for game in self.final_data.keys():
            cache_file = os.path.join(self.data_dir, 'cache', f'{game}.csv')
            if os.path.exists(cache_file) and os.path.isfile(cache_file):
                self.final_data[game] = pd.read_csv(cache_file)
            else:
                return False
        return True

    def save_cache(self):
        if not os.path.exists(os.path.join(self.data_dir, 'cache') or not os.path.isdir(os.path.join(self.data_dir, 'cache'))):
            os.mkdir(os.path.join(self.data_dir, 'cache'))
        for game, data in self.final_data.items():
            data.to_csv(os.path.join(self.data_dir, 'cache', f'{game}.csv'), index=False)

    def postprocess(self, cache=False):
        data_in_each_game = {game: {} for game in self.raw_data['tracking']['gameId'].unique()}
        for game in data_in_each_game:
            data_in_each_game[game] = {
                'pff': self.raw_data['pff'][self.raw_data['pff']['gameId'] == game],
                'tracking': self.raw_data['tracking'][self.raw_data['tracking']['gameId'] == game]
            }
        self.final_data = {game: {} for game in data_in_each_game}
        if cache:
            if self.load_cache():
                return
        for game, data in data_in_each_game.items():
            pff = data['pff'].copy()
            tracking = data['tracking'].copy()
            # create union id by combining nflId and playId for tracking data and pff data
            # remove na for nflId and playId
            tracking = tracking.dropna(subset=['nflId', 'playId'])
            pff = pff.dropna(subset=['nflId', 'playId'])
            tracking['union_id'] = tracking['nflId'].astype(int).astype(str) + '_' + tracking['playId'].astype(int).astype(str)
            pff['union_id'] = pff['nflId'].astype(int).astype(str) + '_' + pff['playId'].astype(int).astype(str)
            # create a new dataframe that concat tracking and pff data columns
            # and copy the pff data to each line which have same union_id, because the pff data is less than tracking data
            # so we need to copy the pff data to each line which have same union_id
            result = tracking.merge(pff, on='union_id', how='left')
            self.final_data[game] = result
        if cache:
            self.save_cache()
        #     # 将 playId和nflid组合成一个新的id，然后判断这个id是否有重复
        #
        #     tracking = data['tracking']
        #     # 1. 选出tracking中的nflId和playId
        #     tracking = tracking[['nflId', 'playId']]
        #     # 2. 将nflId和playId组合成一个新的id
        #     tracking['id'] = tracking['nflId'].astype(str) + '_' + tracking['playId'].astype(str)
        #     # 3. 判断id是否有重复
        #     tracking['is_duplicate'] = tracking.duplicated('id')
        #     tracking['is_nfl_duplicate'] = tracking.duplicated('nflId')
        #     # 4. 选出一组有相同新id的数据demo，然后获取在原先数据中的对应条目
        #     demo_id = tracking[tracking['is_duplicate'] == True].iloc[0]['id']
        #     demo = tracking[tracking['id'] == demo_id]
        #     src_demo = data['tracking'][(data['tracking']['nflId'] == demo.iloc[0]['nflId']) & (data['tracking']['playId'] == demo.iloc[0]['playId'])]
        #     # time example: 2021-09-10T00:26:31.100
        #     first_date = datetime.strptime(src_demo.iloc[0]['time'], '%Y-%m-%dT%H:%M:%S.%f')
        #     last_date = datetime.strptime(src_demo.iloc[-1]['time'], '%Y-%m-%dT%H:%M:%S.%f')
        #     t = (last_date - first_date).total_seconds()
        #     t_list.append(t)
        # print('max time:', max(t_list))
        # print('min time:', min(t_list))
        # print('mean time:', sum(t_list) / len(t_list))

    def statistics(self):
        # statistics target:
        # 1. pff available rate
        # 2. pff_blockType available rate
        # 3. number of games
        # 4. number of data

        number_of_games = len(self.final_data)
        number_of_data = sum([len(self.final_data[game]) for game in self.final_data])
        pff_available = 0
        pff_blockType_available = 0
        total_lines = 0
        for game, data in self.final_data.items():
            total_lines += len(data)
            pff_available += data.dropna(subset=['pff_role']).shape[0]
            pff_blockType_available += data.dropna(subset=['pff_blockType']).shape[0]
        return {
            # 'raw_pff_available_rate': raw_pff_available_rate,
            # 'raw_pff_blockType_available_rate': raw_pff_blockType_available_rate,
            'number_of_games': number_of_games,
            'number_of_data': number_of_data,
            'pff_available_rate': pff_available / total_lines,
            'pff_blockType_available_rate': pff_blockType_available / total_lines
        }

    def get_column(self):
        # return the column dict, key: name, value: type
        return {col: self.final_data[list(self.final_data.keys())[0]][col].dtype for col in self.final_data[list(self.final_data.keys())[0]].columns}

    def get_category(self, category):
        d = []
        for game, data in self.final_data.items():
            d.extend(data[category].unique())
        return list(set(d))

    def __len__(self):
        return sum([len(self.final_data[game]) for game in self.final_data])

    def __getitem__(self, idx):
        return []
