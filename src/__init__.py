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

    @abstractmethod
    def eval(self, evaluation_dataset: EvaluationDataset) -> Evaluation:
        pass
