from typing import Generator, Tuple, Optional
from dataclasses import dataclass

from sklearn.model_selection import KFold

from data import TrainDataset, EvaluationDataset


@dataclass
class Evaluation:
    mae: float
    mse: float
    rmse: float
    accuracy: float  # Balanced accuracy with rounded predictions
    f1: float  # Weighted f1 score with rounded predictions
    train_time: Optional[float] = 0
<<<<<<< HEAD
=======
    max_mem_usage: Optional[float] = 0
    # roc_auc: float
>>>>>>> d7b19130e1bd21d07f28202a26e56e95012d8750

    def __str__(self):
        return "> " + "\n> ".join(f"{k}: {v}" for k, v in self.__dict__.items())


def to_cross_validation_datasets(dataset: TrainDataset,
                                 n_splits: int,
                                 seed: int) -> Generator[Tuple[TrainDataset, EvaluationDataset], None, None]:
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
