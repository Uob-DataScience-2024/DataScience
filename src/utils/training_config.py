import json

import torch

from network.model import SimpleNN
from network import Seq2SeqLSTM, Seq2SeqGRU, SameSizeCNN, TypicalCNN

models = {
    'LSTM': Seq2SeqLSTM,
    'GRU': Seq2SeqGRU,
    'CNN': SameSizeCNN,
    'TypicalCNN': TypicalCNN,
    'SimpleNN': SimpleNN,
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

criterions = {
    'CrossEntropyLoss': torch.nn.CrossEntropyLoss,
    'MSELoss': torch.nn.MSELoss,
    'L1Loss': torch.nn.L1Loss,
    'SmoothL1Loss': torch.nn.SmoothL1Loss
}


class OptimizerHyperparameters(dict):
    lr = 0.001

    def __init__(self, **kwargs):
        kwargs['lr'] = kwargs.get('lr', 0.001)
        super(OptimizerHyperparameters, self).__init__(**kwargs)
        self.update(kwargs)
        for key, value in self.items():
            setattr(self, key, value)


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
        for key, value in self.items():
            setattr(self, key, value)


class RMSpropHyperparameters(OptimizerHyperparameters):
    alpha = 0.99
    eps = 1e-8
    weight_decay = 0
    momentum = 0
    centered = False

    def __init__(self, **kwargs):
        super(RMSpropHyperparameters, self).__init__(**kwargs)
        self.update(kwargs)
        for key, value in self.items():
            setattr(self, key, value)


class SchedulerHyperparameters(dict):
    step_size = 3
    gamma = 0.8

    def __init__(self, **kwargs):
        kwargs['step_size'] = kwargs.get('step_size', 3)
        kwargs['gamma'] = kwargs.get('gamma', 0.8)
        super(SchedulerHyperparameters, self).__init__(**kwargs)
        self.update(kwargs)
        for key, value in self.items():
            setattr(self, key, value)


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
        kwargs['input_dim'] = kwargs.get('input_dim', 14)
        kwargs['hidden_dim'] = kwargs.get('hidden_dim', 256)
        kwargs['num_layers'] = kwargs.get('num_layers', 3)
        kwargs['dropout'] = kwargs.get('dropout', 0.15)
        super(ModelHyperparameters, self).__init__(**kwargs)
        self.update(kwargs)
        for key, value in self.items():
            setattr(self, key, value)


class TrainingHyperparameters(dict):
    learning_rate = 0.001
    batch_size = 2
    num_epochs = 100
    split_ratio = 0.8
    sub_sequence = False
    split_step = 512
    criterion = criterions['MSELoss']

    def __init__(self, **kwargs):
        kwargs['batch_size'] = kwargs.get('batch_size', 2)
        kwargs['num_epochs'] = kwargs.get('num_epochs', 100)
        kwargs['split_ratio'] = kwargs.get('split_ratio', 0.8)
        kwargs['sub_sequence'] = kwargs.get('sub_sequence', False)
        kwargs['split_step'] = kwargs.get('split_step', 512)
        super(TrainingHyperparameters, self).__init__(**kwargs)
        self.update(kwargs)
        for key, value in self.items():
            setattr(self, key, value)
        self.optimizer = optimizers[self.get('optimizer', 'Adam')]
        self.scheduler = schedulers[self.get('scheduler', 'None')]
        self.criterion = criterions[self.get('criterion', 'MSELoss')]
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

        self.update({
            'optimizer': self.get('optimizer', 'Adam'),
            'scheduler': self.get('scheduler', 'None'),
            'criterion': self.get('criterion', 'MSELoss'),
            'optimizer_hyperparameters': self.optimizer_hyperparameters,
            'scheduler_hyperparameters': self.scheduler_hyperparameters
        })


class TrainingConfigure(dict):
    name = 'default'
    input_features = ['playId', 'nflId', 'frameId', 'time', 'jerseyNumber', 'team', 'playDirection', 'x', 'y', 's', 'a', 'dis', 'o', 'dir']
    target_feature = 'pff_role'
    split = True

    def __init__(self, **kwargs):
        kwargs['name'] = kwargs.get('name', 'default')
        kwargs['input_features'] = kwargs.get('input_features', ['playId', 'nflId', 'frameId', 'time', 'jerseyNumber', 'team', 'playDirection', 'x', 'y', 's', 'a', 'dis', 'o', 'dir'])
        kwargs['target_feature'] = kwargs.get('target_feature', 'pff_role')
        kwargs['split'] = kwargs.get('split', True)
        super(TrainingConfigure, self).__init__(**kwargs)
        self.update(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.model = models[self.get('model', 'LSTM')]
        self.model_hyperparameters = ModelHyperparameters(**self.get('model_hyperparameters', {}))
        self.training_hyperparameters = TrainingHyperparameters(**self.get('training_hyperparameters', {}))
        self.update({
            'model': self.get('model', 'LSTM'),
            'model_hyperparameters': self.model_hyperparameters,
            'training_hyperparameters': self.training_hyperparameters
        })

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def to_file(self, path):
        with open(path, 'w') as f:
            json.dump(self, f, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    @staticmethod
    def from_file(path):
        try:
            with open(path, 'r') as f:
                return TrainingConfigure(**json.load(f))
        except:
            data = TrainingConfigure()
            data.to_file(path)
            return data
