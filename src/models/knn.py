import numpy as np

from src.data import TestDataset, TrainDataset
from src import ModelABC


class KNNModel(ModelABC):

    def __init__(self):
        self.users = {}

    def _add_user_rating(self, user_id, product, rating):
        if rating == "rating":
            return
        if user_id not in self.users:
            self.users[user_id] = {product: float(rating[0])}
        else:
            self.users[user_id][product] = float(rating[0])

    @staticmethod
    def cosine_similarity(a, b):
        s = a_sum = b_sum = 0
        for item in {**a, **b}.keys():
            s += a.get(item, 0) * b.get(item, 0)
            a_sum += a.get(item, 0)
            b_sum += b.get(item, 0)
        return s/(a_sum*b_sum)

    def train(self, dataset: TrainDataset):
        # Add all users to user dict
        X = dataset.X.values
        Y = dataset.Y.values

        # Get all user ratings
        for (user_id, item_id, timestamp), rating in zip(X, Y):
            self._add_user_rating(user_id, item_id, rating)
        # users := UserID + ProductID -> rating

        # for item in

        # Calculate similarity Item A, and all other Items
        # Weighted mean of ratings (e.g. 3.5/5) for those items, based on similarity
        #
        # for item in items: SUM (SUM(for user in users: similarity(user, this_user)*Y[Item][user].rating) * similarity(item, this_item))

    def predict(self, dataset: TestDataset):
        results = []
        for (user_a, item_a, timestamp) in dataset.dataset.values:
            unweighted = 0
            total_weights = 0
            for user_b in self.users.keys():
                if item_a in self.users[user_b]:
                    sim = self.cosine_similarity(self.users[user_a], self.users[user_b])
                    unweighted += sim * self.users[user_b][item_a]
                    total_weights += sim
            results.append(unweighted/total_weights)
        return np.array(results)
