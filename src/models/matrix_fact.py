import numpy as np
from numba import njit

from src import TrainDataset, ModelABC, TestDataset, EvaluationDataset, EpochBar


class DictMatrix:

    def __init__(self, dataset: TrainDataset):
        """
        dataset: user id, item id, rating, timestamp

        user id + item id -> rating
        """
        self.users_ratings = []
        self.user_map = {}
        self.item_map = {}

        for i, (user_id, item_id, timestamp, rating) in dataset.dataset.iterrows():

            user_index = self.user_map.setdefault(user_id, len(self.user_map))
            item_index = self.item_map.setdefault(item_id, len(self.item_map))

            # Adding ratings for each (user, item) pair - if the user doesn't exist, it will add it.
            self.users_ratings.append((user_index, item_index, rating))

    def num_users(self):
        return len(self.user_map)

    def num_items(self):
        return len(self.item_map)


class MatrixFactoriser(ModelABC):
    def __init__(self, k: float, hw_init: int):
        self.k = k
        self.hw_init = hw_init
        self.H = self.W = None

    def train(self, dataset: TrainDataset, eval_dataset: EvaluationDataset = None, epochs: int = 1, lr: float = 0.001):
        R = DictMatrix(dataset)

        self.H = np.full((R.num_users(), self.k), self.hw_init)
        self.W = np.full((self.k, R.num_items()), self.hw_init)

        with EpochBar('Processing', max=epochs) as bar:
            for epoch in range(epochs):
                self.train_step(R, self.H, self.W, lr)
                if eval_dataset is not None:
                    print(self.eval(eval_dataset))
                bar.next()

    # @njit(parallel=True)
    @staticmethod
    def train_step(R, H, W, lr):
        for user_index, item_index, rating in R.users_ratings:

            diff = lr * 2 * (rating - H[user_index, :].dot(W[item_index, :]))

            dmse_dh = diff * W[item_index, :]
            dmse_dw = diff * H[user_index, :]

            H[user_index, :] += dmse_dh
            W[user_index, :] += dmse_dw

    def predict(self, dataset: TestDataset) -> np.ndarray:
        predictions = []

        return np.asarray(predictions)
