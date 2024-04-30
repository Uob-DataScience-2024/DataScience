import os
import sys
import time

from ui.special_figs import analysis_ui
from ui.visual import visual_ui

start_import = time.time()

import argparse
import gradio as gr
from loguru import logger

from network.mutable_dataset import DataGenerator
from ui.build_training_ui import nn_ui, rf_ui
from utils.tools import load_data

logger.info(f"Import time: {time.time() - start_import:.2f}s")


def loguru_gradio_handler(record):
    position = record.split('|')[2]
    if position.strip().startswith('actuator.'):
        gr.Info(record)
    return record


logger.add(loguru_gradio_handler, level="DEBUG")

server = None
running = True
exit_signal = False
data_path = "../data"


def on_reload(_data_path):
    global running, data_path
    gr.Warning(f"Reload data from {_data_path}, please wait... and refresh later")
    data_path = _data_path
    running = False
    return data_path


def init_ui(tracking, pff, play, game, player, merge):
    global running, exit_signal
    data_generator = DataGenerator(tracking, pff, play, game, player, merge)
    full_col = merge.game.copy()
    full_col = full_col.merge(merge.player, on='nflId', how='left')
    full_col = full_col.merge(merge.game_info, on='gameId', how='left')
    columns = merge.game.columns.tolist() + merge.player.columns.tolist() + merge.game_info.columns.tolist()
    columns.sort()
    with gr.Blocks("NFL System") as block:
        with gr.Column("Data Loader"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        with gr.Column(scale=10):
                            data_loader = gr.Dropdown(label="Data Loader", choices=["../test_data", "../data"], value=data_path)
                        with gr.Column(scale=2):
                            reload_button = gr.Button("Reload")
                        reload_button.click(on_reload, inputs=[data_loader], outputs=[data_loader])

                with gr.Column(scale=2):
                    pass

        with gr.Tab("Training") as training_tab:
            with gr.Tab("Neural Network") as nn_block:
                nn_ui(columns, data_generator, full_col)
            with gr.Tab("Random Forest") as rf_block:
                rf_ui(columns, data_generator, full_col)
        with gr.Tab("Analysis") as analysis_tab:
            analysis_ui(tracking, pff, play, game, player, merge)
            # with gr.Tab("Statistics") as statistics_block:
            #     pass
        with gr.Tab("Visualization") as visualization_tab:
            with gr.Tab("Video") as video_block:
                visual_ui(tracking, pff, play, game, player, merge)

    return block


# class UI:
#     def __init__(self, data_path, default_weeks=1):
#         self.block = None
#         self.data_path = data_path
#         self.default_weeks = default_weeks
#         tracking, pff, play, game, player, merge = self.load_data(data_path, default_weeks)
#         self.load_ui(tracking, pff, play, game, player, merge)
#
#     def load_data(self, data_path, default_weeks):
#         logger.info("Load init data...")
#         tracking, pff, play, game, player, merge = load_data(data_path, default_weeks)
#         logger.info("Data loaded")
#         return tracking, pff, play, game, player, merge
#
#     def load_ui(self, tracking, pff, play, game, player, merge):
#         logger.info("Init UI")
#         self.block = init_ui(tracking, pff, play, game, player, merge)
#
#     def reload_ui(self, data_path, default_weeks):
#         tracking, pff, play, game, player, merge = self.load_data(data_path, default_weeks)
#         self.block.close()
#         self.load_ui(tracking, pff, play, game, player, merge)
#         self.block.lunch()


def main(args):
    global running, exit_signal, data_path
    while not exit_signal:
        logger.info("Load init data...")
        tracking, pff, play, game, player, merge = load_data(data_path)
        logger.info("Data loaded")
        logger.info(f"Length of all data: {len(merge.game)}")

        logger.info("Init UI")
        block = init_ui(tracking, pff, play, game, player, merge)
        logger.info("Launch UI")
        block.launch(prevent_thread_lock=True, server_name='127.0.0.1')
        time.sleep(1)
        gr.Warning("UI init finished")
        while running:
            time.sleep(1)
        logger.info("UI reload")
        running = True
        block.close()
        logger.info("UI closed")


def parse():
    parser = argparse.ArgumentParser(description='NFL Big Data Bowl')
    parser.add_argument('--data_path', type=str, default='../test_data', help='Path to data folder')
    # parser.add_argument('--default_weeks', type=int, default=1, help='Default weeks to load')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse())
