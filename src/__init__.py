from abc import ABC, abstractmethod
import numpy as np
import sklearn.metrics as metrics
from progress.bar import Bar

from src.data import TrainDataset, TestDataset, EvaluationDataset
from src.evaluation import Evaluation


class ModelABC(ABC):
    @abstractmethod
    def train(self, dataset: TrainDataset, epochs: int = None, lr: float = None):
        pass

    @abstractmethod
    def predict(self, dataset: TestDataset) -> np.ndarray:
        pass

    def eval(self, evaluation_dataset: EvaluationDataset) -> Evaluation:
        test_dataset, y_true = evaluation_dataset.to_test_dataset()
        y_preds = self.predict(test_dataset)

        return Evaluation(
            mse=metrics.mean_squared_error(y_true, y_preds),
        )


class MasterBar(Bar):
    @property
    def remaining_minutes(self):
        return self.eta // 60

    @property
    def remaining_seconds(self):
        return self.eta % 60


class EpochBar(MasterBar):
    message = 'Training'
    fill = '#'
    suffix = '%(index)d / %(max)d - %(remaining_minutes)d:%(remaining_seconds)s eta'


class PercentageBar(MasterBar):
    message = 'Processing'
    fill = '#'
    suffix = '%(percent).1f%% - %(remaining_minutes)d:%(remaining_seconds)s eta'
