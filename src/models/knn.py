from src.data import TestDataset, TrainDataset
from src import ModelABC


class KNNModel(ModelABC):
    def train(self, dataset: TrainDataset):
        pass

    def predict(self, dataset: TestDataset):
        pass
