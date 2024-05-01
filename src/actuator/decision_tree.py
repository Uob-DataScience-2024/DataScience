import io
from typing import Optional, Literal

import numpy as np
from PIL import Image
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold

from ml.machine_learning import load_data
from network.mutable_dataset import DataGenerator
from utils.progress import CallbackProgress

from sklearn.metrics import confusion_matrix, mean_squared_error


class DecisionTreeConfig(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self['n_estimators'] = kwargs.get('n_estimators', 100)
        self['min_samples_split'] = kwargs.get('min_samples_split', 2)
        self['min_samples_leaf'] = kwargs.get('min_samples_leaf', 1)
        self['bootstrap'] = kwargs.get('bootstrap', True)
        self['criterion'] = kwargs.get('criterion', 'gini')
        self['min_impurity_decrease'] = kwargs.get('min_impurity_decrease', 0.0)
        self['oob_score'] = kwargs.get('oob_score', False)


class Trainer:
    def __init__(self, n_estimators: int = 100, min_samples_split: int = 2, min_samples_leaf: int = 1,
                 bootstrap: bool = True, criterion: Literal['gini', 'entropy'] = 'gini', min_impurity_decrease: float = 0.0, oob_score: bool = False):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.min_impurity_decrease = min_impurity_decrease
        self.oob_score = oob_score
        self.model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                            bootstrap=bootstrap, criterion=criterion, min_impurity_decrease=min_impurity_decrease, oob_score=oob_score)

    def train(self, X, Y, split_ratio=0.2):
        logger.info("Splitting data...")
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_ratio)
        sample = Y_test.tolist()
        sample = list(set(sample))
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        logger.info("Start training...")
        self.model.fit(X_train, Y_train)
        logger.info("Training finished")
        Y_pred = self.model.predict(X_test)
        acc = accuracy_score(Y_test, Y_pred)
        logger.info(f"Accuracy: {acc * 100:.2f}%")
        return acc, X_test, Y_test, Y_pred

    def train_kfold(self, X, Y, n_splits, progress):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        accs = []
        task = progress.add_task("Training", total=n_splits)
        for train_index, test_index in kf.split(X):
            self.model = RandomForestClassifier(n_estimators=self.n_estimators, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
                                                bootstrap=self.bootstrap, criterion=self.criterion, min_impurity_decrease=self.min_impurity_decrease, oob_score=self.oob_score)
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            self.model.fit(X_train, Y_train)
            Y_pred = self.model.predict(X_test)
            acc = accuracy_score(Y_test, Y_pred)
            accs.append(acc * 100)
            progress.update(task, advance=1)
            yield accs, X_test, Y_test, Y_pred
        progress.remove_task(task)

    def confusion_matrix(self, X, Y, Y_pred, labels, extra_info=""):
        fig, ax = plt.subplots(figsize=(20, 12))
        cm = confusion_matrix(Y, Y_pred)
        cax = ax.matshow(cm, cmap='Blues')
        fig.colorbar(cax)

        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')

        ax.set_xticks(np.arange(cm.shape[1]))
        ax.set_yticks(np.arange(cm.shape[0]))
        ax.set_xticklabels(np.arange(cm.shape[1]))
        ax.set_yticklabels(np.arange(cm.shape[0]))
        ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5)
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f'{val}', ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=12)  # 增加字体大小
        plt.title(f'Confusion Matrix {"" if extra_info == "" else f"({extra_info})"}')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return Image.open(buf)

    def feature_importance(self, cols, extra_info=""):
        feature_importances = self.model.feature_importances_

        fig = plt.figure(figsize=(16, 8))
        plt.barh(range(len(feature_importances)), feature_importances, align='center')
        plt.yticks(range(len(feature_importances)), cols)
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title(f'Feature Importances in Decision Tree {"" if extra_info == "" else f"({extra_info})"}')
        plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return Image.open(buf)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class DecisionTreeScheduler:
    def __init__(self, dataset_dir, data_generator: DataGenerator = None, config=None, on_new_task=None, on_update=None, on_remove=None):
        self.progress = None
        self.data_type_mapping_inverse = None
        self.mapping_log = None
        self.x_columns = None
        if config is None:
            config = DecisionTreeConfig()
        self.dataset_dir = dataset_dir
        self.data_generator = data_generator
        self.X = None
        self.Y = None
        self.config = config
        self.on_new_task = on_new_task
        self.on_update = on_update
        self.on_remove = on_remove

    def prepare(self, x_columns, y_column, norm=True, player_needed=False, game_needed=False, tracking_data_include=True, pff_data_include=True, drop_all_na=False):
        if self.data_generator is None:
            logger.info("Loading data...")
            tracking, pff, play, game, player, merge = load_data(self.dataset_dir)
            logger.info("Data loaded")
            self.data_generator = DataGenerator(tracking, pff, play, game, player, merge)
        logger.info("Generate and preprocess dataset...")
        self.x_columns = x_columns
        self.y_column = y_column
        self.X, self.Y, self.mapping_log, self.data_type_mapping_inverse = self.data_generator.generate_dataset(x_columns, y_column, data_type='numpy', norm=norm, player_needed=player_needed,
                                                                                                                game_needed=game_needed,
                                                                                                                tracking_data_include=tracking_data_include, pff_data_include=pff_data_include,
                                                                                                                drop_all_na=drop_all_na, with_mapping_log=True)

    def value_remapping(self, d, c):
        if c in self.mapping_log:
            if self.mapping_log[c]['type'] == 'category':
                d = self.mapping_log[c]['mapping'][d]
            if self.mapping_log[c]['type'] == 'numeric':
                d = d * (self.mapping_log[c]['mapping']['max'] - self.mapping_log[c]['mapping']['min']) + self.mapping_log[c]['mapping']['min']
            if self.mapping_log[c]['type'] == 'function':
                d = self.data_type_mapping_inverse[c](d)
        return d

    def train(self, split_ratio=0.2, cross_validation=False, n_splits=5):
        if not cross_validation:
            trainer = Trainer(self.config['n_estimators'], self.config['min_samples_split'], self.config['min_samples_leaf'], self.config['bootstrap'], self.config['criterion'],
                              self.config['min_impurity_decrease'], self.config['oob_score'])
            acc, X_test, Y_test, Y_pred = trainer.train(self.X, self.Y, split_ratio)
            labels = list(map(lambda x: self.value_remapping(x, self.y_column), Y_test))
            conf_matrix = trainer.confusion_matrix(X_test, Y_test, Y_pred, labels)
            importance = trainer.feature_importance(self.x_columns)
            yield f"Accuracy: {acc * 100:.2f}%", [conf_matrix], [importance]
        else:
            self.progress = CallbackProgress(new_progress=self.on_new_task, update=self.on_update, remove_progress=self.on_remove)
            conf_matrixs = []
            importances = []
            with self.progress:
                trainer = Trainer(self.config['n_estimators'], self.config['min_samples_split'], self.config['min_samples_leaf'], self.config['bootstrap'], self.config['criterion'],
                                  self.config['min_impurity_decrease'], self.config['oob_score'])
                for item, X_test, Y_test, Y_pred in trainer.train_kfold(self.X, self.Y, n_splits, self.progress):
                    avg = (sum(item) / len(item)) if len(item) > 0 else 0
                    labels = list(map(lambda x: self.value_remapping(x, self.y_column), Y_test))
                    conf_matrix = trainer.confusion_matrix(X_test, Y_test, Y_pred, labels, extra_info=f"Corss Validation Enabled")
                    importance = trainer.feature_importance(self.x_columns, extra_info=f"Corss Validation Enabled")
                    conf_matrixs.append(conf_matrix)
                    importances.append(importance)
                    yield f"K-Fold result test acc: {avg:.3f}% - " + '/'.join(map(lambda x: f"{x:.3f}%", item)), conf_matrixs, importances
