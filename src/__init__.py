from abc import ABC, abstractmethod
import numpy as np
import sklearn.metrics as metrics

from src.data import TrainDataset, TestDataset, EvaluationDataset
from src.evaluation import Evaluation


class ModelABC(ABC):
    @abstractmethod
    def train(self, dataset: TrainDataset):
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
