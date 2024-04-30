import os

import PIL
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

from actuator.decision_tree import DecisionTreeConfig, DecisionTreeScheduler
from actuator.neural_network import NeuralNetworkScheduler
from network.mutable_dataset import DataGenerator
from ui.tools import ProcessManager, load_config, save_config
from utils import TrainingConfigure
import matplotlib.pyplot as plt
import io

scheduler: NeuralNetworkScheduler = None


def train_nn(
        x_cols: list, y_col: str, num_classes: int,
        model: str, optimizer: str, criterion: str,
        epochs: int, batch_size: int, learning_rate: float, split_ratio: float, k_fold: int,
        gpu: bool = False, norm: bool = False, tracking_data_include: bool = True, pff_data_include: bool = False, player_needed: bool = False, game_needed: bool = False, drop_all_na: bool = False,
        data_generator: DataGenerator = None):
    global scheduler
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
    if scheduler is not None:
        scheduler.close_progress_context()
    scheduler = NeuralNetworkScheduler('../data', 'cuda' if gpu else 'cpu', config, num_classes=num_classes, data_generator=data_generator,
                                       on_new_task=progress_manager.on_new_task, on_update=progress_manager.on_update, on_remove=progress_manager.on_remove)
    scheduler.prepare(norm=norm, tracking_data_include=tracking_data_include, pff_data_include=pff_data_include, player_needed=player_needed, game_needed=game_needed, drop_all_na=drop_all_na)
    for i, (text, image) in enumerate(scheduler.train(epochs, batch_size, split_ratio) if k_fold == 0 else scheduler.train_k_fold(epochs, batch_size, split_ratio, k_folds=k_fold)):
        gr.Info(f"Training {i}/{epochs} epoch...[{(i + 1) / epochs * 100:.2f}%]")
        yield text, *image


def draw_heat_map_for_line(array, title):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(array[np.newaxis, :], cmap='hot', interpolation='nearest')
    ax.set_title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)


def call_feature_analysis(input_cols, limit=10):
    result = scheduler.input_features_analysis(limit)
    result *= 100
    df = pd.DataFrame([list(map(lambda x: f"{x:.3f}%", result.tolist()))], columns=input_cols)
    return df, [draw_heat_map_for_line(result, "Feature Importance")]


def call_feature_analysis_target(input_cols, labels, targets, limit=10):
    if targets is None or len(targets) == 0:
        return call_feature_analysis(input_cols, limit)
    labels.sort()
    results = scheduler.input_features_analysis_with_target(limit, labels, targets)
    data = []
    index = []
    for key, value in results.items():
        data.append(list(map(lambda x: f"{x * 100:.2f}%", value.tolist())))
        index.append(key)
    df = pd.DataFrame(data, columns=input_cols, index=index)
    # copy index to columns
    df['index'] = df.index
    df = df[['index'] + input_cols]
    return df, [draw_heat_map_for_line(value, key) for key, value in results.items()]


def value_remapping(d, c):
    if c in scheduler.data_mapping_log:
        if scheduler.data_mapping_log[c]['type'] == 'category':
            d = scheduler.data_mapping_log[c]['mapping'][d]
        if scheduler.data_mapping_log[c]['type'] == 'numeric':
            d = d * (scheduler.data_mapping_log[c]['mapping']['max'] - scheduler.data_mapping_log[c]['mapping']['min']) + scheduler.data_mapping_log[c]['mapping']['min']
        if scheduler.data_mapping_log[c]['type'] == 'function':
            d = scheduler.data_type_mapping_inverse[c](d)
    return d


def get_data(index, input_cols, y_col):
    indexes = index.split(',')
    indexes = list(map(int, indexes))
    y_list = []
    x_list = []
    for i in indexes:
        x, y = scheduler.dataset[i]
        x = x.numpy().tolist()
        x_new = []
        for d, c in zip(x, input_cols):
            d = value_remapping(d, c)
            x_new.append(d)

        y = value_remapping(y.item(), y_col)
        y_list.append(y)
        x_list.append(x_new)
    df = pd.DataFrame(x_list, columns=input_cols)
    return df, '; '.join(y_list)


def predict_nn(index, labels):
    indexes = index.split(',')
    indexes = list(map(int, indexes))
    labels.sort()
    acc = 0
    results = []
    figs = []
    for i in indexes:
        x, y = scheduler.dataset[i]
        x.to(scheduler.device)
        result = scheduler.predict(x.unsqueeze(0))
        fig = plt.figure()
        plt.bar(labels, (result[0] * 100).tolist())
        pred_index = result.argmax()
        result = list(map(lambda x: f"{x * 100:.3f}", result[0].tolist()))
        results.append(result)
        if pred_index == y.item():
            acc += 1
        plt.title(f"Index: {i}" + (" Correct" if pred_index == y.item() else " Wrong"), color='#38CCB2' if pred_index == y.item() else '#B4424B')
        plt.text(0.5, 1.1, f'Pred: {labels[pred_index]}, Label: {labels[y.item()]}', ha='center', va='bottom', transform=plt.gca().transAxes)
        plt.xlabel("Labels")
        plt.ylabel("Confidence(%)")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        figs.append(Image.open(buf))
    df = pd.DataFrame(results, columns=labels)
    return df, f"Acc: {acc / len(indexes) * 100:.2f}%", figs


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
                tracking_data_include = gr.Checkbox(label="Tracking Data Include", value=False)
                pff_data_include = gr.Checkbox(label="PFF Data Include", value=True)
                player_needed = gr.Checkbox(label="Player Data Needed", value=False)
                game_needed = gr.Checkbox(label="Game Data Needed", value=False)
                drop_all_na = gr.Checkbox(label="Drop All NA", value=False)
        with gr.Column():
            gr.Markdown("### Model settings")
            with gr.Group():
                num_classes = gr.Number(label="Number of Classes", value=2)
                auto_num_classes = gr.Button("Auto Detect Number of Classes")
                auto_num_classes.click(fn=
                                       lambda x:
                                       len(full_col[x].astype(str).unique())
                                       , inputs=[y_col], outputs=[num_classes])
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
                k_fold = gr.Slider(label="K-Fold", value=0, minimum=0, maximum=10, step=1, info="0 means disable k-fold")

    btn_load.click(fn=lambda x: load_config(x), inputs=[cached],
                   outputs=[x_cols, y_col, num_classes, model, optimizer, criterion, epochs, batch_size, learning_rate, split_ratio, gpu, norm, player_needed, game_needed])

    btn_save.click(fn=lambda *x: save_config(config_dir, x),
                   inputs=[x_cols, y_col, num_classes, model, optimizer, criterion, epochs, batch_size, learning_rate, split_ratio, gpu, norm, player_needed, game_needed])

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Train")

            btn_train = gr.Button("Train", variant="primary")
            btn_stop = gr.Button("Stop", variant="stop")
            info = gr.Textbox("Training info", value="")
            with gr.Row():
                image_plot_loss = gr.Plot(label="Training info plot(loss)")
                image_plot_acc = gr.Plot(label="Training info plot(accuracy)")
            # close progress for image plot only
            t_event = btn_train.click(fn=lambda *x: (yield from train_nn(*x, data_generator=data_generator)), show_progress="minimal",
                                      inputs=[x_cols, y_col, num_classes, model, optimizer, criterion, epochs, batch_size, learning_rate, split_ratio, k_fold, gpu, norm, tracking_data_include,
                                              pff_data_include, player_needed, game_needed, drop_all_na],
                                      outputs=[info, image_plot_loss, image_plot_acc])
            btn_stop.click(fn=None, cancels=[t_event])
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Model Analysis")

            gradient_test_limit = gr.Slider(label="Gradient Test Data Limit", value=4, minimum=1, maximum=100, step=1)
            labels_to_test = gr.CheckboxGroup(label="Labels to Test", choices=sorted(full_col['passResult'].astype('category').unique()))
            btn_analysis = gr.Button("Feature Analysis", variant="primary")
            feature_analysis = gr.DataFrame(label="Feature Analysis")
            heat_map = gr.Gallery(label="Heat Map")
            # btn_analysis.click(fn=call_feature_analysis, inputs=[x_cols, gradient_test_limit], outputs=[feature_analysis])
            btn_analysis.click(fn=lambda *x: call_feature_analysis_target(x[0], sorted(full_col[x[1]].astype(str).astype('category').unique()), x[2], x[3]),
                               inputs=[x_cols, y_col, labels_to_test, gradient_test_limit],
                               outputs=[feature_analysis, heat_map])

            def get_labels(col_name):
                sorted(full_col[col_name].astype(str).astype('category').unique())
                return list(filter(lambda x: not pd.isnull(x) and x != 'nan', sorted(full_col[col_name].astype(str).astype('category').unique())))

            y_col.change(lambda x: gr.CheckboxGroup(choices=get_labels(x), value=[]), inputs=[y_col], outputs=[labels_to_test])

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Model Prediction")

            index_of_data = gr.Textbox(label="Index of Data", value="1, 1000, 10000, 114514, 1919810")
            with gr.Row():
                ranges_dataset = gr.Markdown(f"- Dataset Range: 0 - ?")
                btn_get_range = gr.Button("Get Range")
                btn_get_range.click(fn=lambda x: gr.Markdown(f"- Dataset Range: 0 - {len(scheduler.dataset)}"), outputs=[ranges_dataset])
            with gr.Row():
                lmt_rand = gr.Number(label="Random Limit Numbers", value=20)
                index_rand = gr.Button("Random Indexes")
                index_rand.click(fn=lambda x: ', '.join(map(str, np.random.randint(0, len(scheduler.dataset), x).tolist())), inputs=[lmt_rand], outputs=[index_of_data])
            btn_load_detail = gr.Button("Load Detail", variant="primary")
            input_preview = gr.DataFrame(label="Input Preview")
            correct_result = gr.Label(label="Correct Result")
            btn_load_detail.click(fn=get_data, inputs=[index_of_data, x_cols, y_col], outputs=[input_preview, correct_result])
            result_confidence = gr.DataFrame(label="Result Confidence")
            result_acc = gr.Label(label="Result Accuracy")
            btn_predict = gr.Button("Predict", variant="primary")
            confidence_figs = gr.Gallery(label="Confidence Figs")
            btn_predict.click(fn=lambda *x: predict_nn(*x[:1], labels=sorted(full_col[x[1]].astype(str).dropna().astype('category').unique())),
                              inputs=[index_of_data, y_col], outputs=[result_confidence, result_acc, confidence_figs])


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
