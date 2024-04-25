import json
import time

from matplotlib import pyplot as plt

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


def run_visual(gameid, fps, encoder: Literal['opencv', 'ffmpeg'], output_name, time_max, info_config_text, filter_feature, filter_target, tracking, pff, play, game, player, merge):
    visual = Visual(None)
    visual.fps = fps
    visual.encoder = encoder
    visual.output = output_name
    visual.time_max = time_max
    visual.gameid = gameid
    info_config = json.loads(info_config_text)
    for image in visual.run_new_low_memory(gameid, tracking, pff, play, game, player, merge, info_config, filter_feature, filter_target):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        yield image, None

    yield None, visual.output


def on_add_item(col, template, prev_config):
    config = json.loads(prev_config)
    config.append({"col": col, "template": template})
    return json.dumps(config, indent=4), '\n'.join(map(lambda x: f"Column: {x['col']}, Template: \"{x['template']}\"", config)), gr.Dropdown(choices=list(map(lambda x: x['col'], config)))


def on_remove_item(col, prev_config):
    config = json.loads(prev_config)
    config = list(filter(lambda x: x['col'] != col, config))
    return json.dumps(config, indent=4), '\n'.join(map(lambda x: f"Column: {x['col']}, Template: \"{x['template']}\"", config)), gr.Dropdown(choices=list(map(lambda x: x['col'], config)))


def on_filter_feature(feature, target, merge):
    if target is None:
        return None
    df = merge.game
    x = df['gameClock']
    y = df[feature] == target
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(x, y)
    ax.set_xlabel('gameClock')
    ax.set_ylabel(feature)
    ax.set_title(f"{feature} distribution")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


def visual_ui(tracking, pff, play, game, player, merge):
    with gr.Blocks("NFL System") as block:
        gameid = gr.Dropdown(label="Game ID", choices=merge.game['gameId'].unique().tolist(), value=merge.game['gameId'].unique().tolist()[0])
        fps = gr.Slider(label="FPS", minimum=1, maximum=60, value=30)
        encoder = gr.Radio(label="Encoder", choices=["opencv", "ffmpeg"], value="opencv")
        output_name = gr.Textbox(label="Output Path", value="output.mp4")
        time_max = gr.Slider(label="Time Max(Second)", minimum=1, maximum=60 * 60 * 4, value=30)
        time_max_convert_label = gr.Label(label="Time Max(Second) Convert: ")
        time_max.change(lambda x: f"{x // 60}:{x % 60:02d}", inputs=[time_max], outputs=[time_max_convert_label])

        gr.Markdown("## Info display configuration:")
        with gr.Blocks("Info display"):
            info_configer_display = gr.Markdown("", label="Info display configuration: ")
            gr.Markdown("#### Add:")
            with gr.Row():
                info_configer_save = gr.Label("[]", visible=False)
                col_selection = gr.Dropdown(label="Column Selection", choices=merge.game.columns.tolist(), value=None)
                text_template = gr.Textbox(label="Text Template", value="{key}: {value}")
                add_item = gr.Button("Add")
            gr.Markdown("#### Remove:")
            with gr.Row():
                col_able_to_remove = gr.Dropdown(label="Column for remove", choices=[], value=None)
                remove_item = gr.Button("Remove")
                remove_item.click(on_remove_item, inputs=[col_able_to_remove, info_configer_save], outputs=[info_configer_save, info_configer_display, col_able_to_remove])

            add_item.click(on_add_item, inputs=[col_selection, text_template, info_configer_save], outputs=[info_configer_save, info_configer_display, col_able_to_remove])

        gr.Markdown("## Filter configuration:")
        with gr.Blocks("Filter"):
            filter_feature = gr.Dropdown(label="Filter Feature", choices=merge.game.columns.tolist(), value='passResult')
            filter_target = gr.Dropdown(label="Filter Target", choices=merge.game['passResult'].unique().tolist() + [None], value=None)
            # filtered_distribution = gr.Image(label="Filter Distribution")
            filter_feature.change(lambda x: merge.game[x].unique().tolist() + [None], inputs=[filter_feature], outputs=[filter_target])
            # filter_target.change(lambda *x: on_filter_feature(*x, merge), inputs=[filter_feature, filter_target], outputs=[filtered_distribution])

        btn_run = gr.Button("Run")
        preview = gr.Image("Preview", type="numpy", image_mode="RGB", streaming=True)
        output = gr.Video()
        btn_run.click(lambda *x: (yield from run_visual(*x, tracking, pff, play, game, player, merge)),
                      inputs=[gameid, fps, encoder, output_name, time_max, info_configer_save, filter_feature, filter_target], outputs=[preview, output])
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
    tracking, pff, play, game, player, merge = load_data('../data')
    logger.info("Data loaded")

    logger.info("Init UI")
    block = init_ui(tracking, pff, play, game, player, merge)
    logger.info("Launch UI")
    block.launch(server_name='127.0.0.1')


if __name__ == "__main__":
    main()
