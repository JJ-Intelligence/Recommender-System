import numpy as np

from data import TestDataset, TrainDataset, EvaluationDataset
from models import ModelBase


class RandomModel(ModelBase):
    """ Predicts random ratings, modeling ratings using a normal distribution, or just using the average """

    def __init__(self):
        super().__init__()
        self.is_normal = self.rating_mean = self.rating_std = None

    def initialise(self, is_normal: bool, *args, **kwargs):
        self.is_normal = is_normal

    def setup_model(self, dataset: TrainDataset):
        ratings = dataset.dataset["user rating"].to_numpy()
        self.rating_mean = np.mean(ratings)

        if self.is_normal:
            self.rating_std = np.std(ratings)

    def train_step(self, train_dataset: TrainDataset, eval_dataset: EvaluationDataset, *args, **kwargs):
        if self.rating_mean is None:
            self.setup_model(train_dataset)

        if eval_dataset is not None:
            return self.eval(eval_dataset)

    def train(self, train_dataset: TrainDataset, eval_dataset: EvaluationDataset = None, *args, **kwargs):
        """ Note that `epochs` and `lr` have no effect on this model """
        return self.train_step(train_dataset, eval_dataset)

    def predict(self, dataset: TestDataset) -> np.ndarray:
        if self.is_normal:
            return np.random.normal(self.rating_mean, self.rating_std, (len(dataset.dataset),))

        return np.full((len(dataset.dataset),), self.rating_mean)

    def save(self, checkpoint_dir):
        pass

    def load(self, checkpoint_dir):
        pass
