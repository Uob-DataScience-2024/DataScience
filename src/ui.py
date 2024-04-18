import time
start_import = time.time()
import io
from typing import Literal

import cv2
import gradio as gr
import numpy as np
from PIL import Image
from loguru import logger

from network.mutable_dataset import DataGenerator
from ui.build_training_ui import nn_ui, rf_ui
from utils.tools import load_data
from visualization import Visual


logger.info(f"Import time: {time.time() - start_import:.2f}s")
def loguru_gradio_handler(record):
    # 记录包含了所有的日志相关信息
    # 你可以直接访问这些结构化数据
    position = record.split('|')[2]
    if position.strip().startswith('actuator.'):
        gr.Info(record)
    return record


logger.add(loguru_gradio_handler, level="DEBUG")


def run_visual(gameid, fps, encoder: Literal['opencv', 'ffmpeg'], output_name, time_max, targetX, targetY, draw_once, tracking, pff, play, game, player, merge):
    visual = Visual(None)
    visual.fps = fps
    visual.encoder = encoder
    visual.output = output_name
    visual.time_max = time_max
    visual.gameid = gameid
    for image in visual.run_new_low_memory(gameid, tracking, pff, play, game, player, merge, targetX=targetX, targetY=targetY, draw_once=draw_once):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        yield image, None

    yield None, visual.output


def visual_ui(tracking, pff, play, game, player, merge):
    with gr.Blocks("NFL System") as block:
        gameid = gr.Dropdown(label="Game ID", choices=merge.game['gameId'].unique().tolist(), value=merge.game['gameId'].unique().tolist()[0])
        fps = gr.Slider(label="FPS", minimum=1, maximum=60, value=30)
        encoder = gr.Radio(label="Encoder", choices=["opencv", "ffmpeg"], value="opencv")
        output_name = gr.Textbox(label="Output Path", value="output.mp4")
        time_max = gr.Slider(label="Time Max(Second)", minimum=1, maximum=60 * 60 * 4, value=30)

        with gr.Row():
            targetX = gr.Dropdown(label="Target X", choices=merge.game.columns.tolist(), value='jerseyNumber')
            targetY = gr.Dropdown(label="Target Y", choices=merge.game.columns.tolist(), value='pff_role')
        draw_once = gr.Checkbox(label="Draw Once", value=False)

        btn_run = gr.Button("Run")
        preview = gr.Image("Preview", type="numpy", image_mode="RGB", streaming=True)
        output = gr.Video()
        btn_run.click(lambda *x: (yield from run_visual(*x, tracking, pff, play, game, player, merge)),
                      inputs=[gameid, fps, encoder, output_name, time_max, targetX, targetY, draw_once], outputs=[preview, output])
    return block

# def data_filter_cell_ui(col_name, data_type):
#     with gr.Blocks():
#         with gr.Row():
#             with gr.Column():
#                 gr.Markdown(f"### {col_name}")
#                 with gr.Group():
#                     if data_type == 'category':
#                         filter = gr.CheckboxGroup(label="Filter", choices=['All'], value=['All'])
#                     else:
#                         filter = gr.CheckboxGroup(label="Filter", choices=['All', 'None'], value=['All'])
#                 with gr.Group():
#                     if data_type == 'category':
#                         filter_value = gr.CheckboxGroup(label="Filter Value", choices=['All'], value=['All'])
#                     else:
#                         filter_value = gr.CheckboxGroup(label="Filter Value", choices=['All', 'None'], value=['All'])
def base_filter_ui(tracking, pff, play, game, player, merge):
    """
    columns: gameId, team,

    :param tracking:
    :param pff:
    :param play:
    :param game:
    :param player:
    :param merge:
    :return:
    """
    pass

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
            with gr.Tab("Statistics") as statistics_block:
                pass
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
    block.launch(server_name='0.0.0.0')


if __name__ == "__main__":
    main()
