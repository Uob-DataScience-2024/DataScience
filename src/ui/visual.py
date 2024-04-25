import io
import json
from typing import Literal

import cv2
import gradio as gr
from matplotlib import pyplot as plt

from visualization import Visual


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
