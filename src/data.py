import pandas as pd
import numpy as np
from typing import Tuple


class TestDataset:
    """Data from the test file, which we want to generate predictions of"""
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset


class TrainDataset:
    """
    Data from the train file, which we want to learn from
    Split into train/evaluation datasets
    """
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset


class EvaluationDataset(TrainDataset):
    def to_test_dataset(self) -> Tuple[TestDataset, np.ndarray]:
        return TestDataset(self.X), self.y
