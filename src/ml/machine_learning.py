import os

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.utils.data import Dataset
from tqdm.rich import tqdm
from data.total_data import TrackingNormData, PffNormData, PlayNormData, PlayerNormData, GameNormData, MergeNormData
from sklearn.model_selection import train_test_split

from network.model import SimpleNN
from network.mutable_dataset import DataGenerator
from utils.timer import Timer


def load_data(data_dir: str):
    weeks = [x for x in os.listdir(data_dir) if x.startswith('week')]
    if len(weeks) == 0:
        raise ValueError("No week file found")
    weeks = [os.path.join(data_dir, x) for x in weeks]
    pff_file = os.path.join(data_dir, 'pffScoutingData.csv')
    play_file = os.path.join(data_dir, 'plays.csv')
    game_file = os.path.join(data_dir, 'games.csv')
    player_file = os.path.join(data_dir, 'players.csv')
    tracking = TrackingNormData(weeks, parallel_loading=True)
    pff = PffNormData(pff_file)
    play = PlayNormData(play_file)
    game = GameNormData(game_file)
    player = PlayerNormData(player_file)
    merge = MergeNormData(player, game, tracking, pff, play)
    return tracking, pff, play, game, player, merge


def convertTimeToNumerical(t):
    t = t.split(":")
    if len(t) == 2:
        return int(t[0]) * 60 + int(t[1])
    return int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])


def heightInches(h):
    temp = list(map(int, h.split('-')))
    return temp[0] * 12 + temp[1]


def convert_personnelO(text):
    pass


class SimpleDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.long)


def send_to_nural_network(dataset, Y_label):
    # Y = Y.copy()
    # Y_label = Y.unique()
    # Y = Y.astype('category').cat.codes
    # dataset = SimpleDataset(X.values, Y.to_numpy())
    train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def collate_fn(batch):
        x, y = zip(*batch)
        x = torch.stack(x)
        y = torch.stack(y)
        return x.to(device), y.to(device)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1024, shuffle=False, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1024, shuffle=False, collate_fn=collate_fn)
    X, Y = dataset.X, dataset.Y
    model = SimpleNN(X.shape[1], len(Y_label))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        losses = 0
        model.train()
        for i, (x, y) in enumerate(progress := tqdm(train_loader)):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            losses += loss.item()
            optimizer.step()
            progress.set_description(f"Step {i + epoch * len(dataset)} loss: {loss.item()}")
        logger.info(f"Epoch {epoch} loss: {np.mean(losses)}")

        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for x, y in tqdm(test_loader, desc="Testing"):
                y_pred = model(x)
                y_pred = torch.argmax(y_pred, dim=1)
                total += y.size(0)
                correct += (y_pred == y).sum().item()
            logger.info(f"Accuracy: {100 * correct / total:.2f}%")


def main_3():
    logger.info("Starting")
    timer = Timer().start()
    tracking, pff, play, game, player, merge = load_data('../test_data')
    timer.stop().print("Loading", level="debug").reset().start()
    data_generator = DataGenerator(tracking, pff, play, game, player, merge)
    X, Y = data_generator.generate_dataset(['defendersInBox', 'quarter', 'yardsToGo', 'gameClock', 's', 'officialPosition'], 'personnelO', data_type='pandas',
                                           data_type_mapping={'gameClock': convertTimeToNumerical, 'height': heightInches},
                                           norm=True, player_needed=True)
    timer.stop().print("Prepare data", level="debug").reset().start()
    dataset = SimpleDataset(X.values, Y.astype('category').cat.codes.to_numpy())
    send_to_nural_network(dataset, Y.unique())


def main_2():
    logger.info("Starting")
    timer = Timer().start()
    tracking, pff, play, game, player, merge = load_data('../data')
    timer.stop().print("Loading", level="debug").reset().start()
    data_generator = DataGenerator(tracking, pff, play, game, player, merge)
    X, Y = data_generator.generate_dataset(['defendersInBox', 'quarter', 'yardsToGo', 'gameClock', 's', 'officialPosition'], 'passResult', data_type='pandas',
                                           data_type_mapping={'gameClock': convertTimeToNumerical, 'height': heightInches},
                                           norm=True, player_needed=True)
    timer.stop().print("Prepare data", level="debug").reset().start()

    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2)
    timer.stop().print("Split", level="debug").reset().start()

    scaler = MinMaxScaler()
    xTrainScaled = scaler.fit_transform(trainX)
    xTestScaled = scaler.fit_transform(testX)

    timer.stop().print("Scale", level="debug").reset().start()
    model = DecisionTreeClassifier()
    # train model
    model.fit(xTrainScaled, trainY)
    timer.stop().print("Training", level="debug").reset().start()
    predictions = model.predict(xTestScaled)
    timer.stop().print("Predict", level="debug").reset().start()
    # observing performance metrics
    accuracy = accuracy_score(testY, predictions) * 100
    logger.info("Accuracy of the model is {:.2f}%".format(accuracy))


def main():
    logger.info("Starting")
    timer = Timer().start()
    tracking, pff, play, game, player, merge = load_data('../test_data')
    timer.stop().print("Loading", level="debug").reset().start()
    df = merge.game.copy()
    df = df.merge(player.data.copy(), on='nflId', how='left')
    df = df.dropna(subset=['height'])
    df['gameClock'] = df['gameClock'].apply(convertTimeToNumerical)
    df['height'] = df['height'].apply(heightInches)
    df = df.fillna(method='ffill')
    df['officialPosition'] = df['officialPosition'].astype('category').cat.codes
    X = df[['defendersInBox', 'quarter', 'yardsToGo', 'gameClock', 's', 'officialPosition']]
    Y = df['personnelO']
    Y.dropna()
    timer.stop().print("Prepare data", level="debug").reset().start()

    dataset = SimpleDataset(X.values, Y.astype('category').cat.codes.to_numpy())
    data_generator = DataGenerator(tracking, pff, play, game, player, merge)
    dataset_2 = data_generator.generate_dataset(['defendersInBox', 'quarter', 'yardsToGo', 'gameClock', 's', 'officialPosition'], 'personnelO', data_type='torch',
                                                data_type_mapping={'gameClock': convertTimeToNumerical, 'height': heightInches},
                                                norm=False, player_needed=True)
    for i in range(len(dataset)):
        x_0, y_0 = dataset[i]
        x_1, y_1 = dataset_2[i]
        if not torch.allclose(x_0, x_1) or y_0 != y_1:
            logger.error(f"X not equal at index {i}")
    send_to_nural_network(dataset_2, Y.unique())
    #
    # timer.stop().print("NN", level="debug").reset().start()

    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2)
    timer.stop().print("Split", level="debug").reset().start()

    scaler = MinMaxScaler()
    xTrainScaled = scaler.fit_transform(trainX)
    xTestScaled = scaler.fit_transform(testX)

    timer.stop().print("Scale", level="debug").reset().start()
    model = DecisionTreeClassifier()
    # train model
    model.fit(xTrainScaled, trainY)
    timer.stop().print("Training", level="debug").reset().start()
    predictions = model.predict(xTestScaled)
    timer.stop().print("Predict", level="debug").reset().start()
    # observing performance metrics
    accuracy = accuracy_score(testY, predictions) * 100
    logger.info("Accuracy of the model is {:.2f}%".format(accuracy))


if __name__ == '__main__':
    main_2()
