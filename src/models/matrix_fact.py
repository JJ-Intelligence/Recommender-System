import numpy as np

from src import TrainDataset, ModelABC, TestDataset


class DictMatrix:

    def __init__(self, dataset: TrainDataset):
        """
        dataset: user id, item id, rating, timestamp

        user id + item id -> rating
        """
        self.users_ratings = {}
        self.user_map = {}
        self.item_map = {}

        for i, (user_id, item_id, timestamp, rating) in dataset.dataset.iterrows():

            # Adding ratings for each (user, item) pair - if the user doesn't exist, it will add it.
            self.users_ratings.setdefault(user_id, {})[item_id] = rating

            self.user_map.setdefault(user_id, len(self.user_map))
            self.item_map.setdefault(item_id, len(self.item_map))

    def num_users(self):
        return len(self.user_map)

    def num_items(self):
        return len(self.item_map)


class MatrixFactoriser(ModelABC):
    def __init__(self, k: float, pq_init: int):
        self.k = k
        self.pq_init = pq_init

    def train(self, dataset: TrainDataset):
        R = DictMatrix(dataset)

        P = np.full((R.num_users(), self.k), self.pq_init)
        Q = np.full((self.k, R.num_items()), self.pq_init)

    def predict(self, dataset: TestDataset) -> np.ndarray:
        pass
