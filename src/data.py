import numpy as np
import pandas as pd
from typing import Tuple


class TestDataset:
    """Data from the test file, which we want to generate predictions of"""
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset


class TrainDataset:
    """Data from the train file, which we want to learn from"""
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.dataset = pd.concat([self.X, self.y], axis=1)


class EvaluationDataset(TrainDataset):
    """Data from the train file, which will be used to evaluate the performance of a model"""
    def to_test_dataset(self) -> Tuple[TestDataset, np.ndarray]:
        return TestDataset(self.X), self.y.to_numpy()
