import os

import gradio as gr

from actuator.decision_tree import DecisionTreeConfig, DecisionTreeScheduler
from actuator.neural_network import NeuralNetworkScheduler
from network.mutable_dataset import DataGenerator
from ui.tools import ProcessManager, load_config, save_config
from utils import TrainingConfigure


def train_nn(
        x_cols: list, y_col: str, num_classes: int,
        model: str, optimizer: str, criterion: str,
        epochs: int, batch_size: int, learning_rate: float, split_ratio: float,
        gpu: bool = False, norm: bool = False, player_needed: bool = False, game_needed: bool = False,
        data_generator: DataGenerator = None):
    progress_manager = ProcessManager(disable=False)
    config = TrainingConfigure(
        input_features=x_cols, target_feature=y_col,
        model=model, training_hyperparameters={
            'learning_rate': learning_rate, 'batch_size': batch_size, 'split_ratio': split_ratio,
            'criterion': criterion, 'num_epochs': epochs, 'optimizer': optimizer},
        model_hyperparameters={
            'input_dim': len(x_cols), 'hidden_dim': 512, 'num_layers': 3, 'dropout': 0.2
        }
    )
    scheduler = NeuralNetworkScheduler('../data', 'cuda' if gpu else 'cpu', config, num_classes=num_classes, data_generator=data_generator,
                                       on_new_task=progress_manager.on_new_task, on_update=progress_manager.on_update, on_remove=progress_manager.on_remove)
    scheduler.prepare(norm=norm, player_needed=player_needed, game_needed=game_needed)
    for i, (text, image) in enumerate(scheduler.train(epochs, batch_size, split_ratio)):
        gr.Info(f"Training {i}/{epochs} epoch...[{(i + 1) / epochs * 100:.2f}%]")
        yield text, image


def train_rf(
        x_cols: list, y_col: str, split_ratio: float,
        n_estimators: int = 100, min_samples_split: int = 2, min_samples_leaf: int = 1,
        bootstrap: bool = True, criterion: str = 'gini', min_impurity_decrease: float = 0.0, oob_score: bool = False,
        data_generator: DataGenerator = None
):
    config = DecisionTreeConfig(
        n_estimators=n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap, criterion=criterion, min_impurity_decrease=min_impurity_decrease, oob_score=oob_score,

    )
    scheduler = DecisionTreeScheduler('../data', config=config, data_generator=data_generator)
    scheduler.prepare()
    acc = scheduler.train(x_cols, y_col, split_ratio=split_ratio)
    return f"Accuracy: {acc * 100:.2f}%"


def nn_ui(columns, data_generator, full_col, config_dir='configs/nn'):
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    gr.Markdown("## Neural Network Training")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Setting Loader")
            with gr.Row():
                with gr.Column(scale=23):
                    cached = gr.Dropdown(label="Cached Configs", choices=[os.path.join(config_dir, x) for x in os.listdir(config_dir) if x.endswith('.json')])
                with gr.Column(scale=1):
                    refresh = gr.Button("Refresh")
                refresh.click(fn=lambda x: gr.update(x, choices=[os.path.join(config_dir, x) for x in os.listdir(config_dir) if x.endswith('.json')]), outputs=[cached])
            btn_load = gr.Button("Load Config", variant="primary")
            btn_save = gr.Button("Save Config", variant="primary")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Feature settings")
            with gr.Group():
                x_cols = gr.CheckboxGroup(label="X Columns", choices=columns, value=['defendersInBox', 'quarter', 'yardsToGo', 'gameClock', 's', 'officialPosition'])
                y_col = gr.Dropdown(label="Y Column", choices=columns, value='passResult')
                norm = gr.Checkbox(label="Normalize Data", value=False)
                player_needed = gr.Checkbox(label="Player Data Needed", value=False)
                game_needed = gr.Checkbox(label="Game Data Needed", value=False)
        with gr.Column():
            gr.Markdown("### Model settings")
            with gr.Group():
                num_classes = gr.Number(label="Number of Classes", value=2)
                auto_num_classes = gr.Button("Auto Detect Number of Classes")
                auto_num_classes.click(fn=lambda x: len(full_col[x].unique()), inputs=[y_col], outputs=[num_classes])
            gr.Markdown("### Training settings")
            with gr.Group():
                gpu = gr.Checkbox(label="Use GPU", value=False)
                model = gr.Dropdown(label="Model", choices=['SimpleNN'], value='SimpleNN')
                optimizer = gr.Dropdown(label="Optimizer", choices=['Adam', 'SGD', 'RMSprop'], value='Adam')
                criterion = gr.Dropdown(label="Loss function", choices=['MSELoss', 'CrossEntropyLoss'], value='CrossEntropyLoss')
                epochs = gr.Number(label="Epochs", value=10)
                batch_size = gr.Number(label="Batch Size", value=256)
                learning_rate = gr.Slider(label="Learning Rate", minimum=0.0001, maximum=0.1, value=0.001)
                split_ratio = gr.Slider(label="Split Ratio", minimum=0.6, maximum=0.9, value=0.8)

    btn_load.click(fn=lambda x: load_config(x), inputs=[cached],
                   outputs=[x_cols, y_col, num_classes, model, optimizer, criterion, epochs, batch_size, learning_rate, split_ratio, gpu, norm, player_needed, game_needed])

    btn_save.click(fn=lambda *x: save_config(config_dir, x),
                   inputs=[x_cols, y_col, num_classes, model, optimizer, criterion, epochs, batch_size, learning_rate, split_ratio, gpu, norm, player_needed, game_needed])

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Train")

            btn_train = gr.Button("Train", variant="primary")
            # btn_stop = gr.Button("Stop", variant="stop")
            info = gr.Textbox("Training info", value="")
            image_plot = gr.Plot(label="Training info plot")
            # close progress for image plot only
            t_event = btn_train.click(fn=lambda *x: (yield from train_nn(*x[:-1], data_generator=data_generator)), show_progress="minimal",
                                      inputs=[x_cols, y_col, num_classes, model, optimizer, criterion, epochs, batch_size, learning_rate, split_ratio, gpu, norm, player_needed, game_needed,
                                              image_plot],
                                      outputs=[info, image_plot])
            # btn_stop.click(fn=None, cancels=[t_event])


def rf_ui(columns, data_generator, full_col, config_dir='configs/rf'):
    gr.Markdown("## Decision Tree Training")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Setting Loader")
            with gr.Row():
                with gr.Column(scale=23):
                    cached = gr.Dropdown(label="Cached Configs", choices=[os.path.join(config_dir, x) for x in os.listdir(config_dir) if x.endswith('.json')])
                with gr.Column(scale=1):
                    refresh = gr.Button("Refresh")
                refresh.click(fn=lambda x: gr.update(x, choices=[os.path.join(config_dir, x) for x in os.listdir(config_dir) if x.endswith('.json')]), outputs=[cached])
            btn_load = gr.Button("Load Config", variant="primary")
            btn_save = gr.Button("Save Config", variant="primary")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Feature settings")
            with gr.Group():
                x_cols = gr.CheckboxGroup(label="X Columns", choices=columns, value=['defendersInBox', 'quarter', 'yardsToGo', 'gameClock', 's', 'officialPosition'])
                y_col = gr.Dropdown(label="Y Column", choices=columns, value='passResult')
                norm = gr.Checkbox(label="Normalize Data", value=False)
                player_needed = gr.Checkbox(label="Player Data Needed", value=False)
                game_needed = gr.Checkbox(label="Game Data Needed", value=False)
        with gr.Column():
            gr.Markdown("### Fit settings")
            with gr.Group():
                split_ratio = gr.Slider(label="Split Ratio", minimum=0.1, maximum=0.9, value=0.8)
                n_estimators = gr.Number(label="Number of Estimators", value=100)
                min_samples_split = gr.Number(label="Min Samples Split", value=2)
                min_samples_leaf = gr.Number(label="Min Samples Leaf", value=1)
                bootstrap = gr.Checkbox(label="Bootstrap", value=True)
                criterion = gr.Dropdown(label="Criterion", choices=['gini', 'entropy'], value='gini')
                min_impurity_decrease = gr.Number(label="Min Impurity Decrease", value=0.0)
                oob_score = gr.Checkbox(label="OOB Score", value=False)

    btn_load.click(fn=lambda x: load_config(x), inputs=[cached],
                   outputs=[x_cols, y_col, split_ratio, n_estimators, min_samples_split, min_samples_leaf, bootstrap, criterion, min_impurity_decrease, oob_score])
    btn_save.click(fn=lambda *x: save_config(config_dir, x),
                   inputs=[x_cols, y_col, split_ratio, n_estimators, min_samples_split, min_samples_leaf, bootstrap, criterion, min_impurity_decrease, oob_score])
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Train")
            btn_train = gr.Button("Train", variant="primary")
            info = gr.Textbox("Training info", value="")
            t_event = btn_train.click(fn=
                                      lambda *x: train_rf(*x, data_generator=data_generator),
                                      inputs=[x_cols, y_col, split_ratio, n_estimators, min_samples_split, min_samples_leaf, bootstrap, criterion, min_impurity_decrease, oob_score],
                                      outputs=[info])
