from src.data import TestDataset, TrainDataset
from src import ModelABC


class KNNModel(ModelABC):

    def __init__(self):
        self.users = {}

    def _add_user_rating(self, user_id, product, rating):
        if user_id not in self.users:
            self.users[user_id] = {product: rating}
        else:
            self.users[user_id][product] = rating

    def train(self, dataset: TrainDataset):
        # Add all users to user dict
        self.users = {}
        for (i, features), rating in zip(dataset.X.iterrows(), dataset.Y.iterrows()):
            self._add_user_rating(features["user id"], features["item id"], rating)
        print(self.users)

    def predict(self, dataset: TestDataset):
        pass
