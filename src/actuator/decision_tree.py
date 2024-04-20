from typing import Optional, Literal

from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from ml.machine_learning import load_data
from network.mutable_dataset import DataGenerator


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
    def __init__(self, n_estimators: int = 100,  min_samples_split: int = 2, min_samples_leaf: int = 1,
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
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        logger.info("Start training...")
        self.model.fit(X_train, Y_train)
        logger.info("Training finished")
        Y_pred = self.model.predict(X_test)
        return accuracy_score(Y_test, Y_pred)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class DecisionTreeScheduler:
    def __init__(self, dataset_dir, data_generator: DataGenerator = None, config=None):
        if config is None:
            config = DecisionTreeConfig()
        self.dataset_dir = dataset_dir
        self.data_generator = data_generator

    def prepare(self, norm=True, player_needed=False, game_needed=False):
        if self.data_generator is None:
            logger.info("Loading data...")
            tracking, pff, play, game, player, merge = load_data(self.dataset_dir)
            logger.info("Data loaded")
            self.data_generator = DataGenerator(tracking, pff, play, game, player, merge)
        logger.info("Generate and preprocess dataset...")

    def train(self, x_columns, y_column, split_ratio=0.2):
        X, Y = self.data_generator.generate_dataset(x_columns, y_column, data_type='numpy', norm=True, player_needed=True)
        trainer = Trainer()
        return trainer.train(X, Y, split_ratio)
