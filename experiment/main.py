from dataset import DatasetPffBlockType
from model import Seq2Seq
import torch
import torch.utils.data
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hyperparameters_model = {
    'input_dim': 10,
    'hidden_dim': 20,
    'output_dim': 30,
    'batch_first': True
}

hyperparameters_training = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 100,
    'split_ratio': 0.8,
}


def collate_fn(batch):
    pass


def main():
    dataset = DatasetPffBlockType('../data')
    train_set, test_set = torch.utils.data.random_split(dataset,
                                                        [int(len(dataset) * hyperparameters_training['split_ratio']), len(dataset) - int(len(dataset) * hyperparameters_training['split_ratio'])])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=hyperparameters_training['batch_size'], shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=hyperparameters_training['batch_size'], shuffle=True, collate_fn=collate_fn)

    model = Seq2Seq(**hyperparameters_model)


if __name__ == '__main__':
    main()
