import torch
from captum.attr import IntegratedGradients
import random

from loguru import logger
from tqdm import tqdm

from network import SequenceDataset
from model import Seq2SeqGRU
from utils.training_config import TrainingConfigure


def calculate_model_gradient(model, x, label_len, seq_samples=100):
    ig = IntegratedGradients(model)
    attributions = torch.zeros_like(x)
    indexes = random.sample(range(0, x.shape[1]), seq_samples)
    for seq_index in tqdm(indexes, desc='Sampling sequences', total=seq_samples):
        for i in range(0, label_len):
            attributions += ig.attribute(x, target=(seq_index, i))

    attributions /= seq_samples
    return attributions


def calculate_model_gradient_single_label(model, x, seq_samples=100, target_index=0):
    ig = IntegratedGradients(model)
    attributions = torch.zeros_like(x)
    indexes = random.sample(range(0, x.shape[1]), seq_samples)
    for seq_index in tqdm(indexes, desc='Sampling sequences', total=seq_samples):
        attributions += ig.attribute(x, target=(seq_index, target_index))

    attributions /= seq_samples
    return attributions


def visualize_model(model_path, training_config: TrainingConfigure, seq_samples=40):
    dataset = SequenceDataset('../test_data', input_features=training_config.input_features, target_feature=training_config.target_feature, split=training_config.split)
    model_hyperparameters = training_config.model_hyperparameters
    input_dim = model_hyperparameters.input_dim
    hidden_dim = model_hyperparameters.hidden_dim
    num_layers = model_hyperparameters.num_layers
    dropout = model_hyperparameters.dropout
    model = Seq2SeqGRU(input_dim, hidden_dim, dataset.label_size() + 1, num_layers=num_layers, dropout=dropout)

    model.load_state_dict(torch.load(model_path))
    model.cuda()
    X, Y = dataset[0]
    X = X.unsqueeze(0).cuda()
    # X (batch, seq, feature)

    attributions = calculate_model_gradient(model, X, dataset.label_size(), seq_samples=seq_samples)
    # merge sequence to one line
    attributions = attributions.sum(dim=1)
    attributions = torch.softmax(attributions, dim=1)
    for i, feature in enumerate(training_config.input_features):
        logger.info(f"Feature [{i}-{feature}]: {attributions[0][i] * 100 :.2f}%")


def visualize_model_single_label(model_path, training_config: TrainingConfigure, seq_samples=20, label_index=0):
    dataset = SequenceDataset('../test_data', input_features=training_config.input_features, target_feature=training_config.target_feature, split=training_config.split)
    model_hyperparameters = training_config.model_hyperparameters
    input_dim = model_hyperparameters.input_dim
    hidden_dim = model_hyperparameters.hidden_dim
    num_layers = model_hyperparameters.num_layers
    dropout = model_hyperparameters.dropout
    model = Seq2SeqGRU(input_dim, hidden_dim, dataset.label_size() + 1, num_layers=num_layers, dropout=dropout)

    model.load_state_dict(torch.load(model_path))
    model.cuda()
    X, Y = dataset[0]
    X = X.unsqueeze(0).cuda()
    # X (batch, seq, feature)

    logger.info(f"Calculating gradient for label {dataset.label_map[label_index]}")
    attributions = calculate_model_gradient_single_label(model, X, seq_samples=seq_samples, target_index=label_index)
    # merge sequence to one line
    attributions = attributions.sum(dim=1)
    attributions = torch.softmax(attributions, dim=1)
    for i, feature in enumerate(training_config.input_features):
        logger.info(f"Feature [{i}-{feature}]: {attributions[0][i] * 100 :.2f}%")
