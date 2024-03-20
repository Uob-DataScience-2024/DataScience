from loguru import logger
from rich.logging import RichHandler
from tqdm.rich import tqdm

from dataset import DatasetPffBlockType, DatasetPffBlockTypeAutoSpilt
from model import *
import torch
import torch.utils.data
import numpy as np
import warnings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hyperparameters_model = {
    'input_dim': 12,
    'hidden_dim': 256,
    'output_dim': 13,
    'batch_first': True,
    'num_layers': 2,
    'dropout': 0.2,
}

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
    return torch.stack(X).to(device), torch.stack(Y).to(device)


def test(model, criterion, test_loader):
    model.eval()
    accuracies = []
    losses = []
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_loader)):
            output = model(data)
            loss = criterion(output, target)
            losses.append(loss.item())
            output = torch.argmax(output, dim=2)
            target = torch.argmax(target, dim=2)
            accuracy = (output == target).sum().item() / (output.shape[0] * output.shape[1])
            accuracies.append(accuracy)

    logger.info(f'Test Accuracy: {np.mean(accuracies) * 100:.2f}%, Test Loss: {np.mean(losses)}')


def init_transformers():
    input_size = 12
    output_size = 13
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    # BATCH_SIZE = 512
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, input_size, output_size, FFN_HID_DIM)
    return transformer


def main():
    dataset = DatasetPffBlockTypeAutoSpilt('../data', cache=True)
    train_set, test_set = torch.utils.data.random_split(dataset,
                                                        [int(len(dataset) * hyperparameters_training['split_ratio']), len(dataset) - int(len(dataset) * hyperparameters_training['split_ratio'])])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=hyperparameters_training['batch_size'], shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=hyperparameters_training['batch_size'], shuffle=True, collate_fn=collate_fn)

    model = Seq2SeqGRU(**hyperparameters_model)
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters_training['learning_rate'])

    for epoch in range(hyperparameters_training['num_epochs']):
        losses = []
        accuracies = []
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
            accuracy = (output == target).sum().item() / (output.shape[0] * output.shape[1])
            losses.append(loss.item())
            accuracies.append(accuracy)
            progress.set_description_str(
                f'Epoch [{epoch}/{hyperparameters_training["num_epochs"]}], Step [{i}/{len(train_loader)}], Loss: {np.mean(losses)}, Accuracy: {np.mean(accuracies) * 100:.2f}%')
            if i % 5 == 0:
                logger.info(f'Epoch [{epoch}/{hyperparameters_training["num_epochs"]}], Step [{i}/{len(train_loader)}], Loss: {np.mean(losses)}, Accuracy: {np.mean(accuracies) * 100:.2f}%')
            if len(losses) > window:
                losses.pop(0)
                accuracies.pop(0)
        test(model, criterion, test_loader)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    PAD_IDX = 0
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def main_transformer():
    dataset = DatasetPffBlockTypeAutoSpilt('../data', cache=True)
    train_set, test_set = torch.utils.data.random_split(dataset,
                                                        [int(len(dataset) * hyperparameters_training['split_ratio']), len(dataset) - int(len(dataset) * hyperparameters_training['split_ratio'])])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=hyperparameters_training['batch_size'], shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=hyperparameters_training['batch_size'], shuffle=True, collate_fn=collate_fn)

    model = init_transformers()
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters_training['learning_rate'])

    for epoch in range(hyperparameters_training['num_epochs']):
        losses = []
        accuracies = []
        window = 10
        for i, (data, target) in enumerate(progress := tqdm(train_loader)):
            if data.shape[1] >= 20000:
                continue
            data = data.permute(1, 0, 2)
            target = target.permute(1, 0, 2)
            model.train()
            optimizer.zero_grad()
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(data, target)
            output = model(data, target, src_mask, tgt_mask, None, None)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            output = torch.argmax(output, dim=2)
            target = torch.argmax(target, dim=2)
            accuracy = (output == target).sum().item() / (output.shape[0] * output.shape[1])
            losses.append(loss.item())
            accuracies.append(accuracy)
            progress.set_description_str(
                f'Epoch [{epoch}/{hyperparameters_training["num_epochs"]}], Step [{i}/{len(train_loader)}], Loss: {np.mean(losses)}, Accuracy: {np.mean(accuracies) * 100:.2f}%')
            if i % 5 == 0:
                logger.info(f'Epoch [{epoch}/{hyperparameters_training["num_epochs"]}], Step [{i}/{len(train_loader)}], Loss: {np.mean(losses)}, Accuracy: {np.mean(accuracies) * 100:.2f}%')
            if len(losses) > window:
                losses.pop(0)
                accuracies.pop(0)
        # test(model, criterion, test_loader)


if __name__ == '__main__':
    main_transformer()
