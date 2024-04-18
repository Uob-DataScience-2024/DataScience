import datetime
import gc
import os

import numpy as np
import torch
from loguru import logger
from rich.logging import RichHandler
from torch import nn
from tqdm import TqdmExperimentalWarning
import warnings
import torch.utils.data
from network import SequenceDataset, TypicalCNN
from network import Seq2SeqGRU, Seq2SeqLSTM, SameSizeCNN
from tqdm.rich import tqdm
import torch.utils.tensorboard
from rich.progress import Progress

from network.dataset import NonTimeRelatedDataset
from utils import TrainingConfigure
from torch.nn import functional as F

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hyperparameters_training = {
    'learning_rate': 0.001,
    'batch_size': 1,
    'num_epochs': 100,
    'split_ratio': 0.8,
}

SPLIT_STEP = 256


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


def collate_fn_split(batch):
    x, y = collate_fn(batch)
    seq_len = x.shape[1]
    split_plan = [(i, i + SPLIT_STEP) for i in range(0, seq_len, SPLIT_STEP)]
    return x, y, split_plan


def execute_cell(model: torch.nn.Module, x, y, criterion, optimizer=None, encoder_hidden=None, decoder_hidden=None, hidden=False, ignore_na=False):
    if hidden:
        pred_y, (encoder_hidden, decoder_hidden) = model(x, encoder_hidden, decoder_hidden, first=True)
    else:
        pred_y = model(x)
    if ignore_na:
        # y: [batch, seq, feature]
        # na data: feature[0] == 1
        if isinstance(criterion, nn.CrossEntropyLoss):
            y = torch.argmax(y, dim=2)
            loss = criterion(pred_y[y != 0], y[y != 0])
        else:
            loss = criterion(pred_y[y[:, 0] != 1], y[y[:, 0] != 1])
    else:
        if isinstance(criterion, nn.CrossEntropyLoss):
            loss = criterion(pred_y, torch.argmax(y, dim=1))
        else:
            loss = criterion(pred_y, y)
    if optimizer is not None:
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad()

    pred_y = torch.argmax(pred_y, dim=1)
    y = torch.argmax(y, dim=1)
    accuracy_all = (pred_y == y).sum().item() / (pred_y.shape[0])
    accuracy_no_na = (pred_y == y)[y != 0 if ignore_na else y != -1].sum().item() / (pred_y.shape[0])
    return (loss, accuracy_all, accuracy_no_na) if encoder_hidden is None and decoder_hidden is None else (loss, accuracy_all, accuracy_no_na, (encoder_hidden, decoder_hidden))


def test(model, criterion, test_loader, config: TrainingConfigure):
    model.eval()
    accuracies_all = []
    accuracies_no_na = []
    losses = []
    with torch.no_grad():
        with Progress() as progress:
            task_test = progress.add_task('Testing', total=len(test_loader))
            for i, data_group in enumerate(test_loader):
                (data, target) = data_group
                data = data.unsqueeze(1)
                target = target[:, 0, :]
                loss, accuracy_all, accuracy_no_na = execute_cell(model, data, target, criterion)
                losses.append(loss.item())
                accuracies_all.append(accuracy_all)
                if accuracy_no_na != 0:
                    accuracies_no_na.append(accuracy_no_na)
                progress.update(task_test,
                                description=f'Step [{i}/{len(test_loader)}], Loss: {loss.item()}, Accuracy All: {np.mean(accuracies_all) * 100:.2f}% Accuracy No NA: {np.mean(accuracies_no_na) * 100:.2f}%',
                                advance=1)

    logger.info(f'Test Accuracy All: {np.mean(accuracies_all) * 100:.2f}%, Test Accuracy No NA: {np.mean(accuracies_no_na) * 100:.2f}%, Test Loss: {np.mean(losses)}')

    return np.mean(accuracies_all), np.mean(accuracies_no_na), np.mean(losses)


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
    elif config.model == TypicalCNN:
        model = TypicalCNN(dataset.label_size() + 1, hidden_dim, dropout=dropout)
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
    return os.path.join(model_dir, model_name)


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=4.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt是模型对每一类别的预测概率
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def run_task(config: TrainingConfigure, logdir, model_path=None):
    global SPILT_STEP
    SPILT_STEP = config.training_hyperparameters.split_step
    if config.training_hyperparameters.sub_sequence:
        raise ValueError("Sub sequence not implemented")
    dataset = NonTimeRelatedDataset('../test_data', input_features=config.input_features, target_feature=config.target_feature, split=config.split)
    split_ratio = config.training_hyperparameters.split_ratio
    batch_size = config.training_hyperparameters.batch_size
    train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * split_ratio), len(dataset) - int(len(dataset) * split_ratio)])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_split if config.training_hyperparameters.sub_sequence else collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_split if config.training_hyperparameters.sub_sequence else collate_fn)
    if init_model == Seq2SeqGRU or init_model == Seq2SeqLSTM:
        raise ValueError("Seq2Seq model need sub sequence")
    model = init_model(config, dataset)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    # criterion = config.training_hyperparameters.criterion()
    criterion = FocalLoss()
    optimizer = config.training_hyperparameters.optimizer(model.parameters(), lr=config.training_hyperparameters.learning_rate)
    scheduler = config.training_hyperparameters.scheduler(optimizer, **config.training_hyperparameters.scheduler_hyperparameters) if config.training_hyperparameters.scheduler != 'None' else None

    # arguments
    epochs = config.training_hyperparameters.num_epochs

    logger.info("Start training")
    logger.info(f"Input Feature: {config.input_features}, Target Feature: {config.target_feature}")
    logger.info(f"Epochs: {epochs}, Dataset Size: {len(dataset)}, Train Size: {len(train_set)}, Test Size: {len(test_set)}")
    logger.info(f"Input Shape: (batch_size, seq_len, {dataset[0][0].shape[1]}), Output Shape: (batch_size, seq_len, {dataset[0][1].shape[1]})")
    logger.info(
        f"Batch size: {config.training_hyperparameters.batch_size}, Enable Sub Sequence Train: {config.training_hyperparameters.sub_sequence}, Split Step: {config.training_hyperparameters.split_step}")
    logger.info(
        f"Model: {config.model}, Input Size: {config.model_hyperparameters.input_dim}, Hidden Size: {config.model_hyperparameters.hidden_dim}, Num Layers: {config.model_hyperparameters.num_layers}, Dropout: {config.model_hyperparameters.dropout}")
    logger.info(f"Optimizer: {config.training_hyperparameters.optimizer}, Learning Rate: {config.training_hyperparameters.learning_rate}")
    logger.info(f"Scheduler: {config.training_hyperparameters.scheduler}, Scheduler Hyperparameters: {config.training_hyperparameters.scheduler_hyperparameters}")

    # log
    if not os.path.exists(os.path.join(logdir, config.name)):
        os.makedirs(os.path.join(logdir, config.name))
    writer = torch.utils.tensorboard.SummaryWriter(os.path.join(logdir, config.name), filename_suffix=config.name)
    last_acc = 0
    last_acc_na = 0
    last_model = None
    for epoch in range(epochs):
        losses = []
        accuracies_all = []
        accuracies_no_na = []
        window = 5
        with Progress() as progress:
            task_train = progress.add_task('Training', total=len(train_loader))
            for i, data_group in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()
                (data, target) = data_group
                if data.shape[0] <= 1:
                    continue
                data = data.unsqueeze(1)
                target = target[:, 0, :]
                loss, accuracy_all, accuracy_no_na = execute_cell(model, data, target, criterion, optimizer)
                losses.append(loss.item())
                accuracies_all.append(accuracy_all)
                if accuracy_no_na != 0:
                    accuracies_no_na.append(accuracy_no_na)
                else:
                    accuracies_no_na.append(np.mean(accuracies_no_na) if len(accuracies_no_na) > 0 else 0)
                info = f'Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {np.mean(losses)}, Accuracy All: {np.mean(accuracies_all) * 100:.2f}% Accuracy No NA: {np.mean(accuracies_no_na) * 100:.2f}%'
                progress.update(task_train, description=info, advance=1)
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

                torch.cuda.empty_cache()

        if scheduler is not None:
            scheduler.step()
        acc, acc_na, loss = test(model, criterion, test_loader, config)
        writer.add_scalar('Test Accuracy All', acc, epoch)
        writer.add_scalar('Test Accuracy No NA', acc_na, epoch)
        writer.add_scalar('Test Loss', loss, epoch)
        if acc > last_acc or acc_na > last_acc_na:
            logger.info("Saving...")
            if not os.path.exists(os.path.join(logdir, config.name)):
                os.makedirs(os.path.join(logdir, config.name))
            last_model = save_model(model, os.path.join(logdir, config.name), config.name + f'_best({acc * 100:.2f}%)_feature({config.target_feature})')
            logger.info(f"{last_model} Saved")
            last_acc = acc
            last_acc_na = acc_na
        else:
            load_model(model, last_model)

    writer.close()
    del model
    del optimizer
    del criterion
    del scheduler
    del train_loader
    del test_loader
    del train_set
    del test_set
    del dataset
    torch.cuda.empty_cache()


def plan_execute(config_dir='configs', logdir='logdir'):
    configs = [x for x in os.listdir(config_dir) if x.endswith('.json')]
    for i, config in enumerate(configs):
        logger.info(f"Start running {config} ({i}/{len(configs)})")
        run_task(TrainingConfigure.from_file(os.path.join(config_dir, config)), logdir)
        gc.collect()


if __name__ == '__main__':
    print()
    run_task(TrainingConfigure.from_file('example_non_time.json'), 'logdir')
    # plan_execute()
