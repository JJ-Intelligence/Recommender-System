from abc import ABC, abstractmethod

from src.data import TrainDataset, TestDataset, EvaluationDataset
from src.evaluation import Evaluation


class ModelABC(ABC):
    @abstractmethod
    def train(self, dataset: TrainDataset):
        pass

    @abstractmethod
    def predict(self, dataset: TestDataset):
        pass

    def eval(self, evaluation_dataset: EvaluationDataset) -> Evaluation:
        predictions = self.predict(evaluation_dataset.to_test_dataset())


def get_mse(targets, predictions) -> float:
    predictions = model.predict(model.)


def get_f1() -> float:
