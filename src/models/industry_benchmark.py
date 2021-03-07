"""This model is built using Surprise, and is purely for benchmark purposes"""
import numpy as np
from surprise import Reader, Dataset, BaselineOnly

from data import TrainDataset, TestDataset, EvaluationDataset
from models.model_base import ModelBase


class IndustryBaselineModel(ModelBase):
    def __init__(self):
        super().__init__()
        self.model = None

    def initialise(self, *args, **kwargs):
        pass

    def train_step(self, dataset: TrainDataset, eval_dataset: EvaluationDataset, *args, **kwargs):
        pass

    def train(self, _dataset: TrainDataset, **kwargs):
        reader = Reader(rating_scale=(0.5, 5))
        data = Dataset.load_from_df(_dataset.dataset[['user id', 'item id', 'user rating']], reader)
        trainset = data.build_full_trainset()
        self.model = BaselineOnly()
        self.model.fit(trainset)

    def predict(self, dataset: TestDataset) -> np.ndarray:
        return np.array([self.model.estimate(int(u), int(i)) for u, i, t in dataset.dataset.to_numpy()])

    def save(self, checkpoint_dir):
        pass

    def load(self, checkpoint_dir):
        pass
