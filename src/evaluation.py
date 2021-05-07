from typing import Generator, Tuple
from dataclasses import dataclass

from sklearn.model_selection import KFold

from data import TrainDataset, EvaluationDataset


@dataclass
class Evaluation:
    mse: float

    def __str__(self):
        return "\n> ".join(f"{k}: {v}" for k, v in self.__dict__.items())


def to_cross_validation_datasets(dataset: TrainDataset,
                                 n_splits: int,
                                 seed: int) -> Generator[Tuple[TrainDataset, EvaluationDataset]]:
    """ Convert a dataset into a list of cross-validation datasets """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = cv.split(dataset.X, dataset.y)

    for X_train_indices, X_test_indices in splits:
        X_train = dataset.X.iloc[X_train_indices]
        y_train = dataset.y.iloc[X_train_indices]
        train = TrainDataset(X_train, y_train)

        X_test = dataset.X.iloc[X_test_indices]
        y_test = dataset.y.iloc[X_test_indices]
        test = EvaluationDataset(X_test, y_test)
        yield train, test
