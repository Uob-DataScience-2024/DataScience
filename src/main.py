import datetime
import os

import numpy as np
import torch
from loguru import logger
from rich.logging import RichHandler
from tqdm import TqdmExperimentalWarning
import warnings
import torch.utils.data
from dataset import TrackingDataset, SequenceDataset
from model import *
from tqdm.rich import tqdm
import torch.utils.tensorboard

from training_config import TrainingConfigure

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hyperparameters_training = {
    'learning_rate': 0.001,
    'batch_size': 1,
    'num_epochs': 100,
    'split_ratio': 0.8,
}


def collate_fn(batch):
    X = []
    Y = []
    max_seq_len = 0
    for x, y in batch:
        X.append(x)
        Y.append(y)
        max_seq_len = max(max_seq_len, x.shape[0])
    for i in range(len(X)):
        X[i] = np.pad(X[i], ((0, max_seq_len - X[i].shape[0]), (0, 0)))
        Y[i] = np.pad(Y[i], ((0, max_seq_len - Y[i].shape[0]), (0, 0)))
        X[i] = torch.tensor(X[i])
        Y[i] = torch.tensor(Y[i])
    return torch.stack(X).to(device, dtype=torch.float32), torch.stack(Y).to(device, dtype=torch.float32)


def test(model, criterion, test_loader):
    model.eval()
    accuracies_all = []
    accuracies_no_na = []
    losses = []
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_loader)):
            output = model(data)
            loss = criterion(output, target)
            losses.append(loss.item())
            output = torch.argmax(output, dim=2)
            target = torch.argmax(target, dim=2)
            accuracy_all = (output == target).sum().item() / (output.shape[0] * output.shape[1])
            accuracy_no_na = (output == target)[target != 0].sum().item() / (output.shape[0] * output.shape[1])
            accuracies_all.append(accuracy_all)
            accuracies_no_na.append(accuracy_no_na)

    logger.info(f'Test Accuracy All: {np.mean(accuracies_all) * 100:.2f}%, Test Accuracy No NA: {np.mean(accuracies_no_na) * 100:.2f}%, Test Loss: {np.mean(losses)}')

    return np.mean(accuracies_all), np.mean(accuracies_no_na), np.mean(losses)


def main():
    input_features = ['playId', 'nflId', 'frameId', 'time', 'jerseyNumber', 'team', 'playDirection', 'x', 'y', 's', 'a', 'dis', 'o', 'dir']
    dataset = SequenceDataset('../data', input_features=input_features, target_feature='pff_role', split=True)

    train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=2, shuffle=False, collate_fn=collate_fn)

    model = Seq2SeqGRU(14, 256, dataset.label_size() + 1, num_layers=3, dropout=0.15)
    model.to(device)
    logger.info(model)
    x_demo, y_demo = dataset[0]
    logger.info(f'Input Shape: {x_demo.shape}, Output Shape: {y_demo.shape}')
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    for epoch in range(hyperparameters_training["num_epochs"]):
        losses = []
        accuracies_all = []
        accuracies_no_na = []
        window = 10
        for i, (data, target) in enumerate(progress := tqdm(train_loader)):
            model.train()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            output = torch.argmax(output, dim=2)
            target = torch.argmax(target, dim=2)
            accuracy_all = (output == target).sum().item() / (output.shape[0] * output.shape[1])
            accuracy_no_na = (output == target)[target != 0].sum().item() / (output.shape[0] * output.shape[1])
            losses.append(loss.item())
            accuracies_all.append(accuracy_all)
            accuracies_no_na.append(accuracy_no_na)
            progress.set_description_str(
                f'Epoch [{epoch}/{hyperparameters_training["num_epochs"]}], Step [{i}/{len(train_loader)}], Loss: {np.mean(losses)}, Accuracy All: {np.mean(accuracies_all) * 100:.2f}% Accuracy No NA: {np.mean(accuracies_no_na) * 100:.2f}%')
            if i % 5 == 0:
                logger.info(
                    f'Epoch [{epoch}/{hyperparameters_training["num_epochs"]}], Step [{i}/{len(train_loader)}], Loss: {np.mean(losses)}, Accuracy All: {np.mean(accuracies_all) * 100:.2f}% Accuracy No NA: {np.mean(accuracies_no_na) * 100:.2f}%')
            if len(losses) > window:
                losses.pop(0)
                accuracies_all.pop(0)
                accuracies_no_na.pop(0)

        scheduler.step()
        test(model, criterion, test_loader)


def init_model(config: TrainingConfigure, dataset):
    model_hyperparameters = config.model_hyperparameters
    input_dim = model_hyperparameters.input_dim
    hidden_dim = model_hyperparameters.hidden_dim
    num_layers = model_hyperparameters.num_layers
    dropout = model_hyperparameters.dropout

    if config.model == Seq2SeqGRU:
        model = Seq2SeqGRU(input_dim, hidden_dim, dataset.label_size() + 1, num_layers=num_layers, dropout=dropout)
    elif config.model == Seq2SeqLSTM:
        model = Seq2SeqLSTM(input_dim, hidden_dim, dataset.label_size() + 1, num_layers=num_layers, dropout=dropout)
    elif config.model == SameSizeCNN:
        model = SameSizeCNN(input_dim, hidden_dim, dataset.label_size() + 1, num_layers=num_layers, dropout=dropout)
    else:
        raise ValueError('Model not implemented')

    model.to(device)
    return model


def save_model(model, model_dir, model_name):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_name = f"{model_name}_{model.__class__.__name__}_{current_time}.pt"
    torch.save(model.state_dict(), os.path.join(model_dir, model_name))


def run_task(config: TrainingConfigure, logdir):
    dataset = SequenceDataset('../data', input_features=config.input_features, target_feature=config.target_feature, split=config.split)
    split_ratio = config.training_hyperparameters.split_ratio
    train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * split_ratio), len(dataset) - int(len(dataset) * split_ratio)])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=2, shuffle=False, collate_fn=collate_fn)

    model = init_model(config, dataset)
    criterion = config.training_hyperparameters.criterion()
    optimizer = config.training_hyperparameters.optimizer(model.parameters(), lr=config.training_hyperparameters.learning_rate)
    scheduler = config.training_hyperparameters.scheduler(optimizer, **config.training_hyperparameters.scheduler_hyperparameters) if config.training_hyperparameters.scheduler != 'None' else None

    # arguments
    epochs = config.training_hyperparameters.num_epochs

    logger.info("Start training")
    logger.info(f"Epochs: {epochs}, Dataset Size: {len(dataset)}, Train Size: {len(train_set)}, Test Size: {len(test_set)}")
    logger.info(
        f"Model: {config.model}, Input Size: {config.model_hyperparameters.input_dim}, Hidden Size: {config.model_hyperparameters.hidden_dim}, Num Layers: {config.model_hyperparameters.num_layers}, Dropout: {config.model_hyperparameters.dropout}")
    logger.info(f"Optimizer: {config.training_hyperparameters.optimizer}, Learning Rate: {config.training_hyperparameters.learning_rate}")
    logger.info(f"Scheduler: {config.training_hyperparameters.scheduler}, Scheduler Hyperparameters: {config.training_hyperparameters.scheduler_hyperparameters}")

    # log
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = torch.utils.tensorboard.SummaryWriter(logdir)
    last_acc = 0
    for epoch in range(epochs):
        losses = []
        accuracies_all = []
        accuracies_no_na = []
        window = 10
        for i, (data, target) in enumerate(progress := tqdm(train_loader)):
            model.train()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            output = torch.argmax(output, dim=2)
            target = torch.argmax(target, dim=2)
            accuracy_all = (output == target).sum().item() / (output.shape[0] * output.shape[1])
            accuracy_no_na = (output == target)[target != 0].sum().item() / (output.shape[0] * output.shape[1])
            losses.append(loss.item())
            accuracies_all.append(accuracy_all)
            accuracies_no_na.append(accuracy_no_na)
            info = f'Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {np.mean(losses)}, Accuracy All: {np.mean(accuracies_all) * 100:.2f}% Accuracy No NA: {np.mean(accuracies_no_na) * 100:.2f}%'
            progress.set_description_str(info)
            # 需要添加的指标: loss, accuracy, no na acc, learning rate, 对于每个epoch和global step
            writer.add_scalar('Loss', loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Accuracy All', accuracy_all, epoch * len(train_loader) + i)
            writer.add_scalar('Accuracy No NA', accuracy_no_na, epoch * len(train_loader) + i)
            if scheduler is not None:
                writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch * len(train_loader) + i)

            if i % 5 == 0:
                logger.info(info)
            if len(losses) > window:
                losses.pop(0)
                accuracies_all.pop(0)
                accuracies_no_na.pop(0)

        if scheduler is not None:
            scheduler.step()
        acc, acc_na, loss = test(model, criterion, test_loader)
        writer.add_scalar('Test Accuracy All', acc, epoch)
        writer.add_scalar('Test Accuracy No NA', acc_na, epoch)
        writer.add_scalar('Test Loss', loss, epoch)
        if acc > last_acc:
            save_model(model, logdir, config.name + f'_best({acc * 100:.2f}%)')
            last_acc = acc

    writer.close()


if __name__ == '__main__':
    run_task(TrainingConfigure.from_file('example.json'), 'logdir')
