import numpy as np

from data import TestDataset, TrainDataset, EvaluationDataset
from models import ModelBase


class RandomModel(ModelBase):
    """
    Predicts random ratings, using a normal distribution with the average rating as the distribution mean, and the
    distribution standard deviation as a hyper-parameter
    """
    def __init__(self):
        super().__init__()
        self.rating_mean = self.rating_std = None

    def initialise(self, rating_std: float, *args, **kwargs):
        self.rating_std = rating_std

    def setup_model(self, dataset: TrainDataset):
        self.rating_mean = np.mean(dataset.dataset["ratings"].to_numpy())

    def train_step(self, dataset: TrainDataset, eval_dataset: EvaluationDataset, *args, **kwargs):
        if self.rating_mean is None:
            self.setup_model(dataset)

        if eval_dataset is not None:
            return self.eval(eval_dataset)

    def train(self, dataset: TrainDataset, eval_dataset: EvaluationDataset = None, epochs: int = 1, lr: float = 1):
        """ Note that `epochs` and `lr` have no effect on this model """
        return self.train_step(dataset, eval_dataset)

    def predict(self, dataset: TestDataset) -> np.ndarray:
        return np.random.normal(self.rating_mean, self.rating_std, (len(dataset.dataset),))

    def save(self, checkpoint_dir):
        pass

    def load(self, checkpoint_dir):
        pass
