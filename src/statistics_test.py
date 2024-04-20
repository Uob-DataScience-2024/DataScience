import os
import unittest

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from tqdm import tqdm

from data import *
from utils.tools import load_data

data_dir = '../data'
if len(list(filter(lambda x: x.endswith('.csv'), os.listdir(data_dir)))) == 0:
    logger.warning('No csv file found in data directory, now run test data')
    data_dir = '../test_data'


def load_demo_data_nfl_data() -> tuple[dict[int, GameNFLData], GameData, PlayerData]:
    weeks = [x for x in os.listdir(data_dir) if x.startswith('week')]
    if len(weeks) == 0:
        raise ValueError("No week file found")
    weeks = [os.path.join(data_dir, x) for x in weeks]
    pff_file = os.path.join(data_dir, 'pffScoutingData.csv')
    play_file = os.path.join(data_dir, 'plays.csv')
    game_file = os.path.join(data_dir, 'games.csv')
    gameNFLData = GameNFLData.loads(weeks, pff_file, play_file)
    game_data = GameData(game_file)
    player_data = PlayerData(os.path.join(data_dir, 'players.csv'))
    return gameNFLData, game_data, player_data


class StatisticsTest(unittest.TestCase):
    def test_temp(self):
        logger.info("Load init data...")
        tracking, pff, play, game, player, merge = load_data('../data')
        df = merge.game.copy()
        df.merge(merge.player, on='nflId', how='left')
        df.merge(merge.game_info, on='gameId', how='left')
        logger.info("Data loaded")

        df = df[df['gameId'] == 2021090900]
        teams = df['team'].dropna().unique()
        df = df[df['team'] != teams[0]]
        df['gameId-playId'] = df['gameId'].astype(str) + '-' + df['playId'].astype(str)
        df.drop_duplicates(subset=['gameId-playId'], inplace=True)
        plt.hist(df['passResult'].dropna())
        plt.show()

    def test_do(self):
        logger.info("Load init data...")
        tracking, pff, play, game, player, merge = load_data('../data')
        df = merge.game.copy()
        df.merge(merge.player, on='nflId', how='left')
        df.merge(merge.game_info, on='gameId', how='left')
        logger.info("Data loaded")
        teams = tracking.data['team'].dropna().unique()
        df['gameId-playId'] = df['gameId'].astype(str) + '-' + df['playId'].astype(str)
        for team in teams:
            sdf = df[np.logical_and(df['team'] == team, df['defensiveTeam'] == team)]  # the data which is the team's defense
            sdf.drop_duplicates(subset=['gameId-playId'], inplace=True)
            sdf.dropna(subset=['personnelO', 'offenseFormation'], inplace=True)
            dgp = sdf.groupby('offenseFormation')
            labels = [
                sdf['defendersInBox'].dropna().unique(),
                sdf['personnelD'].dropna().unique(),
                sdf['pff_passCoverage'].dropna().unique(),
                sdf['pff_passCoverageType'].dropna().unique()
            ]
            figs = []
            for key, subdata in dgp:
                defendersInBox = subdata['defendersInBox']
                personnelD = subdata['personnelD']
                pff_passCoverage = subdata['pff_passCoverage']
                pff_passCoverageType = subdata['pff_passCoverageType']
                data = [defendersInBox, personnelD, pff_passCoverage, pff_passCoverageType]
                # draw the frequency of each data, but x-axis is the label
                fig, axs = plt.subplots(2, 2, figsize=(12, 6))
                for i, ax in enumerate(axs.flat):
                    ax.hist(data[i], bins=len(labels[i]))
                    ax.set_xticks(labels[i])
                    if type(labels[i][0]) == str and len(labels[i][0]) > 8:
                        ax.set_xticklabels(labels[i], rotation=45)
                    ax.set_title(['defendersInBox', 'personnelD', 'pff_passCoverage', 'pff_passCoverageType'][i])
                fig.suptitle(f'offenseFormation: {key}')
                fig.tight_layout()
                figs.append(fig)

            number_of_figs = len(figs)
            fig_per_row = 3
            side_per_fig = number_of_figs // fig_per_row
            final_fig = plt.figure(figsize=(12 * 4, 6 * 4))
            final_fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
            for i, fig in enumerate(figs):
                ax = final_fig.add_subplot(side_per_fig, fig_per_row, i + 1)
                # fig to image
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.imshow(image)
                # close axis
                plt.axis('off')
            plt.suptitle(f'Team: {team}', fontsize=40)
            plt.tight_layout()
            plt.show()

    def test_df(self):
        logger.info("Load init data...")
        tracking, pff, play, game, player, merge = load_data('../data')
        df = merge.game.copy()
        df.merge(merge.player, on='nflId', how='left')
        df.merge(merge.game_info, on='gameId', how='left')
        logger.info("Data loaded")
        teams = df[df['team'] != 'football']['team'].dropna().unique()
        df['gameId-playId'] = df['gameId'].astype(str) + '-' + df['playId'].astype(str)
        df.drop_duplicates(subset=['gameId-playId'], inplace=True)
        passResult_classes = df['passResult'].dropna().unique()
        result = {

        }
        for team in teams:
            sdf = df[np.logical_and(df['team'] == team, df['defensiveTeam'] == team)].copy()
            sdf.dropna(subset=['personnelO', 'offenseFormation'], inplace=True)
            passResult = sdf['passResult']
            prePenaltyPlayResult = sdf['prePenaltyPlayResult'].mean()
            # count to each classes
            passResultCount = passResult.value_counts()
            result[team] = {
                'passResult': {x: 0 for x in passResult_classes},
                'prePenaltyPlayResult': prePenaltyPlayResult
            }
            for key, value in passResultCount.items():
                result[team]['passResult'][key] = value

        # prePenaltyPlayResult
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)
        dataset = list(map(lambda x: (x[0], x[1]['prePenaltyPlayResult']), result.items()))
        dataset = pd.DataFrame(dataset, columns=['team', 'prePenaltyPlayResult'])
        dataset.dropna()
        dataset.sort_values(by='prePenaltyPlayResult', inplace=True)
        dataset = dataset.to_numpy()
        ax.bar(dataset[:, 0], dataset[:, 1])
        plt.title('average prePenaltyPlayResult by team')
        plt.tight_layout()
        plt.show()

        # passResult
        cols = 3
        rows = 2
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 11, rows * 6))
        dataset = {key: {} for key in passResult_classes}
        for i, (team, data) in enumerate(result.items()):
            for key, value in data['passResult'].items():
                dataset[key][team] = value
        for i, (key, value) in enumerate(dataset.items()):
            subdataset = list(value.items())
            subdataset.sort(key=lambda x: x[1], reverse=True)
            subdataset = np.array(subdataset)
            cyka = []
            for team, count in subdataset:
                for _ in range(int(count)):
                    cyka.append(team)
            ax = axs[i // cols, i % cols]
            ax.hist(cyka, bins=len(teams))
            ax.set_title(key)
        plt.suptitle('passResult by team')
        plt.tight_layout()
        plt.show()

    def test_team(self):
        n_cols = 5
        dpi = 300
        data_year = 2023
        df = pd.read_csv(os.path.join(data_dir, 'plays.csv'))
        result = {}
        formations = df['offenseFormation'].dropna().unique()
        gameNFLData, game_data, player_data = load_demo_data_nfl_data()
        official_position = player_data.df['officialPosition'].dropna().unique()
        pff_role = ["Coverage", "Pass", "Pass block", "Pass route", "Pass rush"]
        home_visitor = game_data.get_home_visitor()
        for game_id, nfl_data in tqdm(gameNFLData.items()):
            df = nfl_data.df
            home, visitor = home_visitor[game_id]
            home_score = df[df['team'] == home]['preSnapHomeScore'].max()
            visitor_score = df[df['team'] == visitor]['preSnapVisitorScore'].max()
            group = df.groupby('team')
            for team, data in group:
                win = home_score > visitor_score if team == home else home_score < visitor_score
                draw = home_score == visitor_score
                players = data['nflId'].dropna().unique().tolist()
                if team not in result:
                    result[team] = {"formations": {x: 0 for x in formations},
                                    "officialPosition": {x: 0 for x in official_position},
                                    "pffRole": {x: 0 for x in pff_role},
                                    "win": 0, "draw": 0, "lose": 0, "players": []}
                for formation in formations:
                    result[team]['formations'][formation] += len(data[data['offenseFormation'] == formation])
                for role in pff_role:
                    result[team]['pffRole'][role] += len(data[data['pff_role'] == role])
                if draw:
                    result[team]['draw'] += 1
                elif win:
                    result[team]['win'] += 1
                else:
                    result[team]['lose'] += 1
                result[team]['players'] += players

        for i, (team, data) in enumerate(result.items()):
            result[team]['officialPosition'] = {}
            temp = player_data.df[np.isin(player_data.df['nflId'], result[team]['players'])]
            officialPosition = temp["officialPosition"].value_counts()
            age = pd.to_datetime(temp['birthDate'].dropna(), format='mixed').apply(lambda x: data_year - x.year).dropna().tolist()
            result[team]['age'] = age
            for position, count in officialPosition.items():
                result[team]['officialPosition'][position] = count

        n_rows = np.ceil(len(result) / n_cols).astype(int)

        # big plot

        # plot formation
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 3))
        for i, (team, data) in enumerate(result.items()):
            ax = axs[i // n_cols, i % n_cols]
            ax.bar(data['formations'].keys(), data['formations'].values())
            ax.set_title(team)
        total_plots = n_rows * n_cols
        for j in range(i + 1, total_plots):
            fig.delaxes(axs[j // n_cols, j % n_cols])
        plt.suptitle('Formations', y=0.99)
        plt.tight_layout()
        plt.show()

        # plot win/lose/draw

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        for i, (team, data) in enumerate(result.items()):
            ax = axs[i // n_cols, i % n_cols]
            ax.bar(['win', 'draw', 'lose'], [data['win'], data['draw'], data['lose']])
            ax.set_title(team)
        total_plots = n_rows * n_cols
        for j in range(i + 1, total_plots):
            fig.delaxes(axs[j // n_cols, j % n_cols])
        plt.suptitle('Win/Draw/Lose', y=0.99)
        plt.tight_layout()
        plt.show()

        # plot official position
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 3))
        for i, (team, data) in enumerate(result.items()):
            ax = axs[i // n_cols, i % n_cols]
            ax.bar(data['officialPosition'].keys(), data['officialPosition'].values())
            ax.set_title(team)
        total_plots = n_rows * n_cols
        for j in range(i + 1, total_plots):
            fig.delaxes(axs[j // n_cols, j % n_cols])
        plt.suptitle('Official Position', y=0.99)
        plt.tight_layout()
        plt.show()

        # plot age
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        for i, (team, data) in enumerate(result.items()):
            ax = axs[i // n_cols, i % n_cols]
            ax.hist(data['age'])
            ax.set_title(team)
        total_plots = n_rows * n_cols
        for j in range(i + 1, total_plots):
            fig.delaxes(axs[j // n_cols, j % n_cols])
        plt.suptitle('Age', y=0.99)
        plt.tight_layout()

        # plot pff role
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 3))
        for i, (team, data) in enumerate(result.items()):
            ax = axs[i // n_cols, i % n_cols]
            ax.bar(data['pffRole'].keys(), data['pffRole'].values())
            ax.set_title(team)
        total_plots = n_rows * n_cols
        for j in range(i + 1, total_plots):
            fig.delaxes(axs[j // n_cols, j % n_cols])
        plt.suptitle('PFF Role', y=0.99)
        plt.tight_layout()
        plt.show()

        # small plot

        # plot players
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.bar(result.keys(), [len(x['players']) for x in result.values()])
        plt.title('Players')
        plt.tight_layout()
        plt.show()

        # plot avg age
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.bar(result.keys(), [np.mean(x['age']) for x in result.values()])
        plt.title('Average Age')
        plt.tight_layout()
        plt.show()

        # plot win rate
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.bar(result.keys(), [x['win'] / (x['win'] + x['lose']) for x in result.values()])
        plt.title('Win Rate')
        plt.tight_layout()
        plt.show()

        self.assertEqual(True, True)
