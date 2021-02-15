from src.data import TestDataSet, TrainDataSet
from src.models import ModelABC


class KNNModel(ModelABC):
    def train(self, dataset: TrainDataSet):
        pass

    def predict(self, dataset: TestDataSet):
        pass
