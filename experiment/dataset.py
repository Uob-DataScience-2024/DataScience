import os.path
from datetime import datetime

from torch.utils.data import Dataset
import pandas as pd


class DatasetPffBlockType(Dataset):
    def __init__(self, data_dir, pff_filename='pffScoutingData.csv', tracking_filename='week{}.csv'):
        self.data_dir = data_dir
        self.pff_filename = pff_filename
        self.tracking_filename = tracking_filename
        self.raw_data = {
            'pff': self.load_pff_data(),
            'tracking': self.load_tracking_data()
        }
        self.postprocess()

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

    def postprocess(self):
        data_in_each_game = {game: {} for game in self.raw_data['tracking']['gameId'].unique()}
        for game in data_in_each_game:
            data_in_each_game[game] = {
                'pff': self.raw_data['pff'][self.raw_data['pff']['gameId'] == game],
                'tracking': self.raw_data['tracking'][self.raw_data['tracking']['gameId'] == game]
            }
        final_data = {game: {} for game in data_in_each_game}
        t_list = []
        for game, data in data_in_each_game.items():
            # 将 playId和nflid组合成一个新的id，然后判断这个id是否有重复

            tracking = data['tracking']
            # 1. 选出tracking中的nflId和playId
            tracking = tracking[['nflId', 'playId']]
            # 2. 将nflId和playId组合成一个新的id
            tracking['id'] = tracking['nflId'].astype(str) + '_' + tracking['playId'].astype(str)
            # 3. 判断id是否有重复
            tracking['is_duplicate'] = tracking.duplicated('id')
            tracking['is_nfl_duplicate'] = tracking.duplicated('nflId')
            # 4. 选出一组有相同新id的数据demo，然后获取在原先数据中的对应条目
            demo_id = tracking[tracking['is_duplicate'] == True].iloc[0]['id']
            demo = tracking[tracking['id'] == demo_id]
            src_demo = data['tracking'][(data['tracking']['nflId'] == demo.iloc[0]['nflId']) & (data['tracking']['playId'] == demo.iloc[0]['playId'])]
            # time example: 2021-09-10T00:26:31.100
            first_date = datetime.strptime(src_demo.iloc[0]['time'], '%Y-%m-%dT%H:%M:%S.%f')
            last_date = datetime.strptime(src_demo.iloc[-1]['time'], '%Y-%m-%dT%H:%M:%S.%f')
            t = (last_date - first_date).total_seconds()
            t_list.append(t)
        print('max time:', max(t_list))
        print('min time:', min(t_list))
        print('mean time:', sum(t_list) / len(t_list))

