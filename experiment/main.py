from loguru import logger
from rich.logging import RichHandler
from tqdm.rich import tqdm

from dataset import DatasetPffBlockType
from model import Seq2Seq
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
    'hidden_dim': 512,
    'output_dim': 13,
    'batch_first': True,
    'num_layers': 1,
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


def main():
    dataset = DatasetPffBlockType('../data', cache=True)
    train_set, test_set = torch.utils.data.random_split(dataset,
                                                        [int(len(dataset) * hyperparameters_training['split_ratio']), len(dataset) - int(len(dataset) * hyperparameters_training['split_ratio'])])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=hyperparameters_training['batch_size'], shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=hyperparameters_training['batch_size'], shuffle=True, collate_fn=collate_fn)

    model = Seq2Seq(**hyperparameters_model)
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


if __name__ == '__main__':
    main()
