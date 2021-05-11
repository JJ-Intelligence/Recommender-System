"""This model is built using Surprise, and is purely for benchmark purposes"""
import numpy as np
from surprise import Reader, Dataset, KNNBasic, SVD, NormalPredictor, SlopeOne, NMF, PredictionImpossible, KNNBaseline, KNNWithMeans

from data import TrainDataset, TestDataset, EvaluationDataset
from models.model_base import ModelBase


class KNNBenchmark(ModelBase):
    def __init__(self):
        super().__init__()
        self.model = None

    def initialise(self, knn_class=None, k=40, *args, **kwargs):
        self.knn_class = knn_class
        self.k = k

    def train_step(self, dataset: TrainDataset, eval_dataset: EvaluationDataset, *args, **kwargs):
        pass

    def train(self, _dataset: TrainDataset, **kwargs):
        reader = Reader(rating_scale=(0.5, 5))
        data = Dataset.load_from_df(_dataset.dataset[['user id', 'item id', 'user rating']], reader)
        trainset = data.build_full_trainset()
        model = {
            "KNNBasic": KNNBasic,
            "KNNBaseline": KNNBaseline
        }[self.knn_class]
        self.model = model(verbose=True, sim_options={'name': 'cosine', 'user_based': False}, k=self.k)
        self.model.fit(trainset)
        ratings = _dataset.dataset["user rating"].to_numpy()
        self.rating_mean = np.mean(ratings)

    def predict(self, dataset: TestDataset) -> np.ndarray:
        r = []
        for u, i, t in dataset.dataset.to_numpy():
            try:
                r.append(self.model.predict(int(u), int(i)).est)
            except PredictionImpossible:
                r.append(self.rating_mean)  # Global avg for cold start
        return np.array(r)

    def save(self, checkpoint_dir):
        pass

    def load(self, checkpoint_dir):
        pass


class SVDBenchmark(ModelBase):
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
        self.model = SVD(verbose=True)
        self.model.fit(trainset)
        ratings = _dataset.dataset["user rating"].to_numpy()
        self.rating_mean = np.mean(ratings)

    def predict(self, dataset: TestDataset) -> np.ndarray:
        r = []
        for u, i, t in dataset.dataset.to_numpy():
            try:
                r.append(self.model.predict(int(u), int(i)).est)
            except PredictionImpossible:
                r.append(self.rating_mean)  # Global avg for cold start
        return np.array(r)

    def save(self, checkpoint_dir):
        pass

    def load(self, checkpoint_dir):
        pass