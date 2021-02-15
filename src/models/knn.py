from src.data import TestDataset, TrainDataset
from src import ModelABC, EvaluationDataset


class KNNModel(ModelABC):
    def train(self, dataset: TrainDataset):
        pass

    def predict(self, dataset: TestDataset):
        pass

    def eval(self, evaluation_dataset: EvaluationDataset):
        pass
