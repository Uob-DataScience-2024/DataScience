import time

from ui.special_figs import analysis_ui
from ui.visual import visual_ui

start_import = time.time()

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


def init_ui(tracking, pff, play, game, player, merge):
    data_generator = DataGenerator(tracking, pff, play, game, player, merge)
    full_col = merge.game.copy()
    full_col = full_col.merge(merge.player, on='nflId', how='left')
    full_col = full_col.merge(merge.game_info, on='gameId', how='left')
    columns = merge.game.columns.tolist() + merge.player.columns.tolist() + merge.game_info.columns.tolist()
    columns.sort()
    with gr.Blocks("NFL System") as block:
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


def main():
    logger.info("Load init data...")
    tracking, pff, play, game, player, merge = load_data('../test_data')
    logger.info("Data loaded")

    logger.info("Init UI")
    block = init_ui(tracking, pff, play, game, player, merge)
    logger.info("Launch UI")
    block.launch(server_name='127.0.0.1')


if __name__ == "__main__":
    main()
