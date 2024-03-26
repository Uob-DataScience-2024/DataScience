import json

import torch

from model import Seq2SeqLSTM, Seq2SeqGRU, SameSizeCNN

models = {
    'LSTM': Seq2SeqLSTM,
    'GRU': Seq2SeqGRU,
    'CNN': SameSizeCNN
}

optimizers = {
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD,
    'RMSprop': torch.optim.RMSprop,
}

schedulers = {
    'None': 'None',
    'StepLR': torch.optim.lr_scheduler.StepLR,
    'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'CyclicLR': torch.optim.lr_scheduler.CyclicLR,
}


class OptimizerHyperparameters(dict):
    lr = 0.001

    def __init__(self, **kwargs):
        super(OptimizerHyperparameters, self).__init__(**kwargs)
        self.update(kwargs)


class AdamHyperparameters(OptimizerHyperparameters):
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 0
    amsgrad = False

    def __init__(self, **kwargs):
        super(AdamHyperparameters, self).__init__(**kwargs)
        self.update(kwargs)


class SGDHyperparameters(OptimizerHyperparameters):
    momentum = 0.9
    dampening = 0
    weight_decay = 0
    nesterov = False

    def __init__(self, **kwargs):
        super(SGDHyperparameters, self).__init__(**kwargs)
        self.update(kwargs)


class RMSpropHyperparameters(OptimizerHyperparameters):
    alpha = 0.99
    eps = 1e-8
    weight_decay = 0
    momentum = 0
    centered = False

    def __init__(self, **kwargs):
        super(RMSpropHyperparameters, self).__init__(**kwargs)
        self.update(kwargs)


class SchedulerHyperparameters(dict):
    step_size = 3
    gamma = 0.8

    def __init__(self, **kwargs):
        super(SchedulerHyperparameters, self).__init__(**kwargs)
        self.update(kwargs)


class StepLRHyperparameters(SchedulerHyperparameters):
    def __init__(self, **kwargs):
        super(StepLRHyperparameters, self).__init__(**kwargs)
        self.update(kwargs)


class ReduceLROnPlateauHyperparameters(SchedulerHyperparameters):
    def __init__(self, **kwargs):
        super(ReduceLROnPlateauHyperparameters, self).__init__(**kwargs)
        self.update(kwargs)


class CyclicLRHyperparameters(SchedulerHyperparameters):
    def __init__(self, **kwargs):
        super(CyclicLRHyperparameters, self).__init__(**kwargs)
        self.update(kwargs)


class ModelHyperparameters(dict):
    input_dim = 14
    hidden_dim = 256
    num_layers = 3
    dropout = 0.15

    def __init__(self, **kwargs):
        super(ModelHyperparameters, self).__init__(**kwargs)
        self.update(kwargs)


class TrainingHyperparameters(dict):
    learning_rate = 0.001
    batch_size = 2
    num_epochs = 100
    split_ratio = 0.8

    def __init__(self, **kwargs):
        super(TrainingHyperparameters, self).__init__(**kwargs)
        self.update(kwargs)
        self.optimizer = optimizers[self.get('optimizer', 'Adam')]
        self.scheduler = schedulers[self.get('scheduler', 'None')]
        if self.optimizer == torch.optim.Adam:
            self.optimizer_hyperparameters = AdamHyperparameters(**self.get('optimizer_hyperparameters', {}))
        elif self.optimizer == torch.optim.SGD:
            self.optimizer_hyperparameters = SGDHyperparameters(**self.get('optimizer_hyperparameters', {}))
        elif self.optimizer == torch.optim.RMSprop:
            self.optimizer_hyperparameters = RMSpropHyperparameters(**self.get('optimizer_hyperparameters', {}))
        else:
            self.optimizer_hyperparameters = OptimizerHyperparameters(**self.get('optimizer_hyperparameters', {}))

        if self.scheduler == torch.optim.lr_scheduler.StepLR:
            self.scheduler_hyperparameters = SchedulerHyperparameters(**self.get('scheduler_hyperparameters', {}))
        elif self.scheduler == torch.optim.lr_scheduler.ReduceLROnPlateau:
            self.scheduler_hyperparameters = ReduceLROnPlateauHyperparameters(**self.get('scheduler_hyperparameters', {}))
        elif self.scheduler == torch.optim.lr_scheduler.CyclicLR:
            self.scheduler_hyperparameters = CyclicLRHyperparameters(**self.get('scheduler_hyperparameters', {}))
        else:
            self.scheduler_hyperparameters = SchedulerHyperparameters(**self.get('scheduler_hyperparameters', {}))


class TrainingConfigure(dict):
    name = 'default'

    def __init__(self, **kwargs):
        super(TrainingConfigure, self).__init__(**kwargs)
        self.update(kwargs)
        self.model = models[self.get('model', 'LSTM')]
        self.model_hyperparameters = ModelHyperparameters(**self.get('model_hyperparameters', {}))
        self.training_hyperparameters = TrainingHyperparameters(**self.get('training_hyperparameters', {}))
