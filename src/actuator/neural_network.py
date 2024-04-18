import numpy as np
import torch
from loguru import logger
from torch import nn
from rich.progress import Progress
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from utils.progress import CallbackProgress
from utils.tools import load_data
from utils.training_config import ModelHyperparameters, OptimizerHyperparameters, TrainingConfigure
from network.mutable_dataset import SimpleDataset, DataGenerator
from matplotlib import pyplot as plt


class Trainer:
    def __init__(self, model: nn.Module, optimizer, criterion, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.writer = SummaryWriter()
        self.device = device

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

    def train(self, train_loader, test_loader, epochs, progress: Progress, display_window=10, same_mode=True, regression_task=False, regression_allow_diff=5):
        progress_epoch = progress.add_task("[yellow]Epochs...", total=epochs)
        loss_epoch = []
        accuracy_epoch = []
        for epoch in range(epochs):
            train_loss, train_accuracy = self.train_epoch(train_loader, epoch, progress, display_window, same_mode, regression_task, regression_allow_diff)
            test_loss, test_accuracy = self.test_epoch(test_loader, progress, display_window, same_mode, regression_task, regression_allow_diff)
            progress.update(progress_epoch, advance=1, description=f"Epoch {epoch + 1}/{epochs}, " +
                                                                   f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.4f}%, " +
                                                                   f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.4f}%")
            loss_epoch.append(test_loss)
            accuracy_epoch.append(test_accuracy)
            self.writer.add_scalar("Training/Epoch Loss", train_loss, epoch)
            self.writer.add_scalar("Training/Epoch Accuracy", train_accuracy * 100, epoch)
            self.writer.add_scalar("Testing/Epoch Loss", test_loss, epoch)
            self.writer.add_scalar("Testing/Epoch Accuracy", test_accuracy * 100, epoch)
            yield (f"Epoch {epoch + 1}/{epochs}, " +
                   f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.4f}%, " +
                   f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.4f}%"), self.draw_plot(loss_epoch, accuracy_epoch)

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

    def evaluate(self, test_loader, progress: Progress = None, display_window=10, same_mode=True, regression_task=False, regression_allow_diff=5):
        if progress is None:
            progress = Progress()
        test_loss, test_accuracy = self.test_epoch(test_loader, progress, display_window, same_mode, regression_task, regression_allow_diff)
        progress.print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.4f}%")
        progress.print()


class NeuralNetworkScheduler:
    def __init__(self, dataset_dir, device, config: TrainingConfigure, num_classes=2, data_generator: DataGenerator = None, on_new_task=None, on_update=None, on_remove=None):
        self.dataset_dir = dataset_dir
        self.data_generator = data_generator
        self.config = config
        self.model = config.model(**config.model_hyperparameters, num_classes=num_classes)
        self.model = self.model.to(device)
        self.optimizer = config.training_hyperparameters.optimizer(self.model.parameters(), **config.training_hyperparameters.optimizer_hyperparameters)
        # self.scheduler = config.training_hyperparameters.scheduler(self.optimizer, **config.training_hyperparameters.scheduler_hyperparameters)
        self.criterion = config.training_hyperparameters.criterion()
        self.dataset = None
        self.trainner = Trainer(self.model, self.optimizer, self.criterion, device=device)
        self.ready = False
        self.device = device
        self.on_new_task = on_new_task
        self.on_update = on_update
        self.on_remove = on_remove

    def prepare(self, norm=True, player_needed=False, game_needed=False):
        if self.data_generator is None:
            logger.info("Loading data...")
            tracking, pff, play, game, player, merge = load_data(self.dataset_dir)
            logger.info("Data loaded")
            self.data_generator = DataGenerator(tracking, pff, play, game, player, merge)
        logger.info("Generate and preprocess dataset...")
        # TODO: add stored data_type_mapping
        columns_x = self.config.input_features
        column_y = self.config.target_feature
        self.dataset = self.data_generator.generate_dataset(columns_x, column_y, data_type='torch', player_needed=player_needed, game_needed=game_needed, norm=norm)
        logger.info("Dataset ready")
        self.ready = True

    def train(self, epochs, batch_size, split_ratio, display_window=10, regression_task=False, regression_allow_diff=5):
        if not self.ready:
            raise ValueError("Dataset not ready")
        logger.info("Splitting dataset...")
        train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [int(len(self.dataset) * split_ratio), len(self.dataset) - int(len(self.dataset) * split_ratio)])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        logger.info("Start training...")
        with CallbackProgress(new_progress=self.on_new_task, update=self.on_update, remove_progress=self.on_remove) as progress:
            yield from self.trainner.train(train_loader, test_loader, epochs, progress, display_window=display_window, regression_task=regression_task, regression_allow_diff=regression_allow_diff)
