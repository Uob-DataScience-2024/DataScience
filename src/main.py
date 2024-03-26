import numpy as np
import torch
from loguru import logger
from rich.logging import RichHandler
from tqdm import TqdmExperimentalWarning
import warnings
import torch.utils.data
from dataset import TrackingDataset
from model import LSTMClassifier
from tqdm.rich import tqdm

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
        X[i] = torch.tensor(X[i])
        # Y[i] = torch.tensor(Y[i])
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
            output = torch.argmax(output, dim=1)
            target = torch.argmax(target, dim=1)
            accuracy = (output == target).sum().item() / (output.shape[0])
            accuracies.append(accuracy)

    logger.info(f'Test Accuracy: {np.mean(accuracies) * 100:.2f}%, Test Loss: {np.mean(losses)}')


def main():
    dataset = TrackingDataset('../data')

    train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = LSTMClassifier(input_size=54, hidden_size=256, num_layers=2, num_classes=2)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    for epoch in range(hyperparameters_training["num_epochs"]):
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
            output = torch.argmax(output, dim=1)
            target = torch.argmax(target, dim=1)
            accuracy = (output == target).sum().item() / (output.shape[0])
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
