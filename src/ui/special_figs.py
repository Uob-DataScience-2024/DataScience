import io
from typing import Literal

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image
from loguru import logger
from matplotlib import pyplot as plt


def plt_to_image(fig, result_type: Literal['pil', 'bytes', 'ndarray'] = 'pil'):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    if result_type == 'pil':
        return image
    elif result_type == 'bytes':
        return buf.getvalue()
    elif result_type == 'ndarray':
        return np.array(image)


def draw_defender_info(team, group_col, tracking, pff, play, game, player, merge):
    df = merge.game.copy()

    sdf = df[np.logical_and(df['team'] == team, df['defensiveTeam'] == team)]  # the data which is the team's defense
    sdf['gameId-playId'] = sdf['gameId'].astype(str) + '-' + sdf['playId'].astype(str)
    sdf.drop_duplicates(subset=['gameId-playId'], inplace=True)
    sdf.dropna(subset=[group_col], inplace=True)
    dgp = sdf.groupby(group_col)
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
        fig, axs = plt.subplots(2, 2, figsize=(14, 6))
        for i, ax in enumerate(axs.flat):
            ax.hist(data[i], bins=len(labels[i]))
            ax.set_xticks(labels[i])
            if type(labels[i][0]) == str and len(labels[i][0]) > 8:
                ax.set_xticklabels(labels[i], rotation=30)
            ax.set_title(['defendersInBox', 'personnelD', 'pff_passCoverage', 'pff_passCoverageType'][i])
        fig.suptitle(f'team: {team} - {group_col}: {key}')
        fig.tight_layout()
        figs.append(fig)
    # if len(figs) < 8:
    #     for _ in range(8 - len(figs)):
    #         figs.append(None)
    return map(plt_to_image, figs)


def draw_team_two_info(targets, tracking, pff, play, game, player, merge):
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

    figs = []

    if 'prePenaltyPlayResult' in targets:
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
        figs.append(fig)

    if 'passResult' in targets:
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
        figs.append(fig)

    return map(plt_to_image, figs)


def draw_team_info(targets, tracking, pff, play, game, player, merge):
    n_cols = 5
    data_year = 2023
    df = merge.game.copy()
    result = {}
    formations = df['offenseFormation'].dropna().unique()
    # gameNFLData, game_data, player_data = load_demo_data_nfl_data()
    official_position = player.data['officialPosition'].dropna().unique()
    pff_role = ["Coverage", "Pass", "Pass block", "Pass route", "Pass rush"]
    home_visitor = game.data.groupby('gameId')[['homeTeamAbbr', 'visitorTeamAbbr']].first().to_dict()
    group_df = df.groupby('gameId')
    for game_id, sub_df in group_df:
        home, visitor = home_visitor['homeTeamAbbr'][game_id], home_visitor['visitorTeamAbbr'][game_id]
        sub_df = sub_df[sub_df['team'] != 'football']
        sub_df['gameId-playId'] = sub_df['gameId'].astype(str) + '-' + sub_df['playId'].astype(str)
        sub_df.drop_duplicates(subset=['gameId-playId'], inplace=True)
        home_score = sub_df[sub_df['team'] == home]['preSnapHomeScore'].max()
        visitor_score = sub_df[sub_df['team'] == visitor]['preSnapVisitorScore'].max()
        group = sub_df.groupby('team')
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
        temp = player.data.copy()
        temp = temp[np.isin(temp['nflId'], result[team]['players'])]
        officialPosition = temp["officialPosition"].value_counts()
        age = pd.to_datetime(temp['birthDate'].dropna(), format='mixed').apply(lambda x: data_year - x.year).dropna().tolist()
        result[team]['age'] = age
        for position, count in officialPosition.items():
            result[team]['officialPosition'][position] = count

    n_rows = np.ceil(len(result) / n_cols).astype(int)

    # big plot
    figs = []
    # plot formation
    if 'formations' in targets:
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
        figs.append(fig)

    # plot win/lose/draw
    if 'win/lose/draw' in targets:
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
        figs.append(fig)

    # plot official position
    if 'official position' in targets:
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
        figs.append(fig)

    # plot age
    if 'age' in targets:
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
        figs.append(fig)

    # plot pff role
    if 'pff role' in targets:
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
        figs.append(fig)

    # small plot

    # plot players
    if 'players' in targets:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.bar(result.keys(), [len(x['players']) for x in result.values()])
        plt.title('Players')
        plt.tight_layout()
        figs.append(fig)

    # plot avg age
    if 'AverageAge' in targets:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.bar(result.keys(), [np.mean(x['age']) for x in result.values()])
        plt.title('Average Age')
        plt.tight_layout()
        figs.append(fig)

    # plot win rate
    if 'WinRate' in targets:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.bar(result.keys(), [x['win'] / (x['win'] + x['lose']) for x in result.values()])
        plt.title('Win Rate')
        plt.tight_layout()
        figs.append(fig)

    return map(plt_to_image, figs)


def defender_info_ui(teams, tracking, pff, play, game, player, merge):
    gr.Markdown("## Defender Info")
    team = gr.Dropdown(choices=teams.tolist(), label="Team")
    group_condition = gr.Dropdown(choices=['offenseFormation', 'personnelO'], label="Group Condition")
    btn_plot = gr.Button("Plot")
    gallery = gr.Gallery(label="Plots")

    btn_plot.click(lambda *x: draw_defender_info(*x, tracking, pff, play, game, player, merge), inputs=[team, group_condition], outputs=gallery)


def team_statistics_ui(teams, tracking, pff, play, game, player, merge):
    gr.Markdown("## Team Statistics")
    targets = gr.CheckboxGroup(choices=['prePenaltyPlayResult', 'passResult'], label="Targets")
    btn_plot = gr.Button("Plot")
    gallery = gr.Gallery(label="Plots")

    btn_plot.click(lambda *x: draw_team_two_info(*x, tracking, pff, play, game, player, merge), inputs=[targets], outputs=gallery)


def team_info_ui(teams, tracking, pff, play, game, player, merge):
    gr.Markdown("## Team Info")
    targets = gr.CheckboxGroup(choices=['formations', 'win/lose/draw', 'official position', 'age', 'pff role', 'players', 'AverageAge', 'WinRate'], label="Targets")
    btn_plot = gr.Button("Plot")
    gallery = gr.Gallery(label="Plots")

    btn_plot.click(lambda *x: draw_team_info(*x, tracking, pff, play, game, player, merge), inputs=[targets], outputs=gallery)


def analysis_ui(tracking, pff, play, game, player, merge):
    teams = merge.game['team'].dropna().unique()
    teams = teams[teams != 'football']
    teams.sort()
    with gr.Tab("Defender Info"):
        with gr.Row():
            with gr.Column():
                defender_info_ui(teams, tracking, pff, play, game, player, merge)
    with gr.Tab("Two Type of Bar Chat Info"):
        with gr.Row():
            with gr.Column():
                team_statistics_ui(teams, tracking, pff, play, game, player, merge)
    with gr.Tab("Team Info"):
        with gr.Row():
            with gr.Column():
                team_info_ui(teams, tracking, pff, play, game, player, merge)
