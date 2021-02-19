from src.data import TestDataset, TrainDataset
from src import ModelABC


class KNNModel(ModelABC):
    def train(self, dataset: TrainDataset, epochs: int = None, lr: float = None):
        pass

    def predict(self, dataset: TestDataset):
        pass
