import numpy as np
import torch
from captum.attr import IntegratedGradients
from loguru import logger
from torch import nn
from rich.progress import Progress
import torch.utils.data
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.progress import CallbackProgress
from utils.tools import load_data
from utils.training_config import ModelHyperparameters, OptimizerHyperparameters, TrainingConfigure
from network.mutable_dataset import SimpleDataset, DataGenerator
from matplotlib import pyplot as plt
import cache

class Trainer:
    def __init__(self, model: nn.Module, optimizer, criterion, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.writer = SummaryWriter()
        self.device = device
        self.test_loss_pool = []
        self.test_accuracy_pool = []
        self.train_loss_pool = []
        self.train_accuracy_pool = []

    def train_step(self, x, y, same_mode=True, regression_task=False, regression_allow_diff=5):
        self.optimizer.zero_grad()
        pred_y = self.model(x)
        # if not same_mode:
        #     y = torch.argmax(y, dim=1)
        loss = self.criterion(pred_y, y)
        loss.backward()
        self.optimizer.step()
        if not regression_task:
            pred_y = torch.argmax(pred_y, dim=1)
            # if same_mode:
            #     y = torch.argmax(y, dim=1)
            correct = (pred_y == y).sum().item()
            accuracy = correct / len(y)
        else:
            diff = regression_allow_diff - (pred_y - y).detach().cpu().numpy()
            accuracy = np.mean(diff / regression_allow_diff) if regression_allow_diff != 0 else 0
        return loss.item(), accuracy

    def test_step(self, x, y, same_mode=True, regression_task=False, regression_allow_diff=5):
        pred_y = self.model(x)
        # if not same_mode:
        #     y = torch.argmax(y, dim=1)
        loss = self.criterion(pred_y, y)
        if not regression_task:
            pred_y = torch.argmax(pred_y, dim=1)
            # if same_mode:
            #     y = torch.argmax(y, dim=1)
            correct = (pred_y == y).sum().item()
            accuracy = correct / len(y)
        else:
            diff = regression_allow_diff - (pred_y - y).detach().cpu().numpy()
            accuracy = np.mean(diff / regression_allow_diff) if regression_allow_diff != 0 else 0
        return loss.item(), accuracy

    def train_epoch(self, train_loader, current_epoch, progress: Progress, display_window=10, same_mode=True, regression_task=False, regression_allow_diff=5):
        self.model.train()
        total_loss = []
        total_accuracy = []
        train_progress = progress.add_task("[green]Training...", total=len(train_loader))
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            loss, accuracy = self.train_step(x, y, same_mode, regression_task, regression_allow_diff)
            total_loss.append(loss)
            total_accuracy.append(accuracy)
            progress.update(train_progress, advance=1,
                            description=f"Loss: {loss:.4f}, Accuracy: {accuracy * 100:.4f}%, " +
                                        f"Mean Loss: {np.mean(total_loss[display_window:]):.4f}, Mean Accuracy: {np.mean(total_accuracy[display_window:]) * 100:.4f}%")
            self.writer.add_scalar("Training/Loss", loss, current_epoch * len(train_loader) + i)
            self.writer.add_scalar("Training/Accuracy", accuracy * 100, current_epoch * len(train_loader) + i)
        progress.remove_task(train_progress)
        return np.mean(total_loss), np.mean(total_accuracy)

    def test_epoch(self, test_loader, progress: Progress, display_window=10, same_mode=True, regression_task=False, regression_allow_diff=5):
        self.model.eval()
        total_loss = []
        total_accuracy = []
        test_progress = progress.add_task("[blue]Testing...", total=len(test_loader))
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(self.device), y.to(self.device)
                loss, accuracy = self.test_step(x, y, same_mode, regression_task, regression_allow_diff)
                total_loss.append(loss)
                total_accuracy.append(accuracy)
                progress.update(test_progress, advance=1,
                                description=f"Loss: {loss:.4f}, Accuracy: {accuracy * 100:.4f}%, " +
                                            f"Mean Loss: {np.mean(total_loss[display_window:]):.4f}, Mean Accuracy: {np.mean(total_accuracy[display_window:]) * 100:.4f}%")
                self.writer.add_scalar("Testing/Loss", loss, len(test_loader) + i)
                self.writer.add_scalar("Testing/Accuracy", accuracy * 100, len(test_loader) + i)
        progress.remove_task(test_progress)
        return np.mean(total_loss), np.mean(total_accuracy)

    def train(self, train_loader, test_loader, epochs, progress: Progress, display_window=10, same_mode=True, regression_task=False, regression_allow_diff=5, cross_val=False):
        progress_epoch = progress.add_task("[yellow]Epochs...", total=epochs)
        loss_epoch = []
        accuracy_epoch = []

        for i, epoch in enumerate(range(epochs)):
            train_loss, train_accuracy = self.train_epoch(train_loader, epoch, progress, display_window, same_mode, regression_task, regression_allow_diff)
            test_loss, test_accuracy = self.test_epoch(test_loader, progress, display_window, same_mode, regression_task, regression_allow_diff)
            progress.update(progress_epoch, advance=1, description=f"Epoch {epoch + 1}/{epochs}, " +
                                                                   f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.4f}%, " +
                                                                   f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.4f}%")
            loss_epoch.append(test_loss)
            accuracy_epoch.append(test_accuracy)
            # if cross_val:
            #     self.train_loss_pool.append(train_loss)
            #     self.train_accuracy_pool.append(train_accuracy)
            #     self.test_loss_pool.append(test_loss)
            #     self.test_accuracy_pool.append(test_accuracy)
            #     train_loss = np.mean(self.train_loss_pool)
            #     train_accuracy = np.mean(self.train_accuracy_pool)
            #     test_loss = np.mean(self.test_loss_pool)
            #     test_accuracy = np.mean(self.test_accuracy_pool)
            self.writer.add_scalar("Training/Epoch Loss", train_loss, epoch)
            self.writer.add_scalar("Training/Epoch Accuracy", train_accuracy * 100, epoch)
            self.writer.add_scalar("Testing/Epoch Loss", test_loss, epoch)
            self.writer.add_scalar("Testing/Epoch Accuracy", test_accuracy * 100, epoch)
            if cross_val and epochs - 1 == i:
                self.train_loss_pool.append(train_loss)
                self.train_accuracy_pool.append(train_accuracy)
                self.test_loss_pool.append(test_loss)
                self.test_accuracy_pool.append(test_accuracy)
            yield (f"Epoch {epoch + 1}/{epochs}, " +
                   f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.4f}%, " +
                   f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.4f}%" + (
                       "" if not cross_val else (
                               f"\n K-Fold Cross Validation: \n" + f"Train Loss: {np.mean(self.train_loss_pool)} | Train Accuracy: {np.mean(self.train_accuracy_pool) * 100:.4f}% \n" +
                               f"Test Loss: {np.mean(self.test_loss_pool)} | Test Accuracy: {np.mean(self.test_accuracy_pool) * 100:.4f}% \n" +
                               f"loss std: {np.std(test_loss)} | acc std: {np.std(test_accuracy)} [Cross Validation Enabled]"))), self.draw_plot_2(loss_epoch, accuracy_epoch)

    @staticmethod
    def draw_plot(loss, acc):
        fig, ax = plt.subplots(2)
        ax[0].plot(loss)
        ax[0].set_title('Loss')
        ax[0].grid()
        ax[1].plot(np.array(acc) * 100)
        ax[1].set_title('Accuracy')
        ax[1].grid()
        fig.tight_layout()
        return fig

    @staticmethod
    def draw_plot_2(loss, acc):
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(loss, label='Loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend(loc='upper left')
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(np.array(acc) * 100, label='Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend(loc='upper left')
        return fig1, fig2

    def evaluate(self, test_loader, progress: Progress = None, display_window=10, same_mode=True, regression_task=False, regression_allow_diff=5):
        if progress is None:
            progress = Progress()
        test_loss, test_accuracy = self.test_epoch(test_loader, progress, display_window, same_mode, regression_task, regression_allow_diff)
        progress.print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.4f}%")
        progress.print()


class NeuralNetworkScheduler:
    def __init__(self, dataset_dir, device, config: TrainingConfigure, num_classes=2, data_generator: DataGenerator = None, on_new_task=None, on_update=None, on_remove=None):
        self.progress = None
        self.test_loader = None
        self.train_loader = None
        self.dataset_dir = dataset_dir
        self.data_generator = data_generator
        self.config = config
        self.num_classes = num_classes
        self.model = config.model(**config.model_hyperparameters, num_classes=self.num_classes)
        self.model = self.model.to(device)
        self.optimizer = config.training_hyperparameters.optimizer(self.model.parameters(), **config.training_hyperparameters.optimizer_hyperparameters)
        # self.scheduler = config.training_hyperparameters.scheduler(self.optimizer, **config.training_hyperparameters.scheduler_hyperparameters)
        self.criterion = config.training_hyperparameters.criterion()
        self.dataset = None
        self.data_mapping_log = None
        self.data_type_mapping_inverse = None
        self.trainner = Trainer(self.model, self.optimizer, self.criterion, device=device)
        self.ready = False
        self.device = device
        self.on_new_task = on_new_task
        self.on_update = on_update
        self.on_remove = on_remove

    def re_init(self):
        self.model = self.config.model(**self.config.model_hyperparameters, num_classes=self.num_classes)
        self.model = self.model.to(self.device)
        self.optimizer = self.config.training_hyperparameters.optimizer(self.model.parameters(), **self.config.training_hyperparameters.optimizer_hyperparameters)
        self.trainner.model = self.model
        self.trainner.optimizer = self.optimizer

    def prepare(self, norm=True, tracking_data_include=True, pff_data_include=True, player_needed=False, game_needed=False, drop_all_na=False):
        if self.data_generator is None:
            logger.info("Loading data...")
            tracking, pff, play, game, player, merge = load_data(self.dataset_dir)
            logger.info("Data loaded")
            self.data_generator = DataGenerator(tracking, pff, play, game, player, merge)
        logger.info("Generate and preprocess dataset...")
        # TODO: add stored data_type_mapping
        columns_x = self.config.input_features
        column_y = self.config.target_feature
        self.dataset, self.data_mapping_log, self.data_type_mapping_inverse = self.data_generator.generate_dataset(columns_x, column_y, data_type='torch', player_needed=player_needed,
                                                                                                                   game_needed=game_needed, norm=norm, tracking_data_include=tracking_data_include,
                                                                                                                   pff_data_include=pff_data_include, with_mapping_log=True, drop_all_na=drop_all_na)
        logger.info("Dataset ready")
        logger.warning(f"Dataset size: {len(self.dataset)}")
        self.ready = True

    def train(self, epochs, batch_size, split_ratio, display_window=10, regression_task=False, regression_allow_diff=5):
        if not self.ready:
            raise ValueError("Dataset not ready")
        logger.info("Splitting dataset...")
        train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [int(len(self.dataset) * split_ratio), len(self.dataset) - int(len(self.dataset) * split_ratio)])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.train_loader = train_loader
        self.test_loader = test_loader
        logger.info("Start training...")
        # self.progress = CallbackProgress(new_progress=self.on_new_task, update=self.on_update, remove_progress=self.on_remove)
        self.progress = cache.progress
        with self.progress:
            yield from self.trainner.train(train_loader, test_loader, epochs, self.progress, display_window=display_window, regression_task=regression_task,
                                           regression_allow_diff=regression_allow_diff)

    def train_k_fold(self, epochs, batch_size, split_ratio, display_window=10, k_folds=5, regression_task=False, regression_allow_diff=5):
        if not self.ready:
            raise ValueError("Dataset not ready")
        logger.info("Splitting dataset...")
        splits = self.k_fold_cross_val_split(self.dataset, k_folds=k_folds)
        logger.info("Start training...")
        # self.progress = CallbackProgress(new_progress=self.on_new_task, update=self.on_update, remove_progress=self.on_remove)
        self.progress = cache.progress
        with self.progress:
            task_k_fold = self.progress.add_task(f"K-Fold Cross Validation", total=k_folds)
            for fold in range(k_folds):
                train_idx = [idx for i, sublist in enumerate(splits) if i != fold for idx in sublist]
                test_idx = splits[fold]

                train_subsampler = Subset(self.dataset, train_idx)
                test_subsampler = Subset(self.dataset, test_idx)

                self.train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=False)
                self.test_loader = DataLoader(test_subsampler, batch_size=batch_size, shuffle=False)
                self.re_init()
                yield from self.trainner.train(self.train_loader, self.test_loader, epochs, self.progress, display_window=display_window, regression_task=regression_task,
                                               regression_allow_diff=regression_allow_diff, cross_val=True)
                self.progress.update(task_k_fold, advance=1)

    def close_progress_context(self):
        if self.progress is not None:
            self.progress.stop()

    @staticmethod
    def k_fold_cross_val_split(dataset, k_folds=5, shuffle=True):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))

        if shuffle:
            torch.manual_seed(42)
            indices = torch.randperm(dataset_size).tolist()

        fold_sizes = [dataset_size // k_folds for _ in range(k_folds)]
        for i in range(dataset_size % k_folds):
            fold_sizes[i] += 1

        current = 0
        splits = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            splits.append(indices[start:stop])
            current = stop

        return splits

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            x = x.to(self.device)
            # 对每个类别计算置信度:
            y = self.model(x)
            y = torch.softmax(y, dim=1)
            return y.detach().cpu().numpy()

    def input_features_analysis(self, limit=10):
        self.model.train()
        ig = IntegratedGradients(self.model)
        attributions = []
        for i, (x, y) in enumerate(self.test_loader):
            if i > limit:
                break
            x, y = x.to(self.device), y.to(self.device)
            # target = torch.argmax(y, dim=1)
            attributions += ig.attribute(x, target=y)
        # take mean of all attributions
        attributions = torch.stack(attributions).mean(dim=0)
        attributions = torch.softmax(attributions, dim=0)
        return attributions.detach().cpu().numpy()

    def input_features_analysis_with_target(self, limit, labels, targets):
        self.model.train()
        ig = IntegratedGradients(self.model)
        attributions = {target: [] for target in targets}
        for i, (x, y) in enumerate(self.test_loader):
            if i > limit:
                break
            x, y = x.to(self.device), y.to(self.device)
            # target = torch.argmax(y, dim=1)
            for target in targets:
                idx = labels.index(target)
                y = torch.tensor([idx] * len(y)).to(self.device)
                attributions[target] += ig.attribute(x, target=y)
        # take mean of all attributions
        for att, value in attributions.items():
            value = torch.stack(value).mean(dim=0)
            value = torch.softmax(value, dim=0)
            attributions[att] = value.detach().cpu().numpy()
        return attributions
