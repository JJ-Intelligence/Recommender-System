from abc import ABC, abstractmethod

from src.data import TrainDataSet, TestDataSet


class ModelABC(ABC):
    @abstractmethod
    def train(self, dataset: TrainDataSet):
        pass

    @abstractmethod
    def predict(self, dataset: TestDataSet):
        pass
