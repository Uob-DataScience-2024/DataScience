from typing import Optional, Literal

from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold

from ml.machine_learning import load_data
from network.mutable_dataset import DataGenerator
from utils.progress import CallbackProgress


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
        return acc

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
            yield accs
        progress.remove_task(task)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class DecisionTreeScheduler:
    def __init__(self, dataset_dir, data_generator: DataGenerator = None, config=None, on_new_task=None, on_update=None, on_remove=None):
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
        self.X, self.Y = self.data_generator.generate_dataset(x_columns, y_column, data_type='numpy', norm=norm, player_needed=player_needed, game_needed=game_needed,
                                                              tracking_data_include=tracking_data_include, pff_data_include=pff_data_include, drop_all_na=drop_all_na)

    def train(self, split_ratio=0.2, cross_validation=False, n_splits=5):
        with CallbackProgress(new_progress=self.on_new_task, update=self.on_update, remove_progress=self.on_remove) as progress:
            if not cross_validation:
                trainer = Trainer()
                yield f"Accuracy: {trainer.train(self.X, self.Y, split_ratio) * 100:.2f}%"
            else:
                trainer = Trainer()
                for item in trainer.train_kfold(self.X, self.Y, n_splits, progress):
                    avg = (sum(item) / len(item)) if len(item) > 0 else 0
                    yield f"K-Fold result test acc: {avg:.3f}% - " + '/'.join(map(lambda x: f"{x:.3f}%", item))
