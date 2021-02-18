import numpy as np

from src import TrainDataset


class MatrixMapper:
    def __init__(self, dataset: TrainDataset):
        """
        dataset: user id, item id, rating, timestamp

        user id + item id -> rating
        """
        self.users = {}

        for i, (user_id, item_id, timestamp, rating) in dataset.dataset.iterrows():
            if user_id not in self.users:
                self.users[user_id] = {item_id: rating}
            else:
                self.users[user_id][item_id] = rating

    def get(self, user_id, item_id):
        if user_id in self.users:
            return self.users.get(user_id).get(item_id, 0)
        return 0


def lazy_dot(P, Q):
    """Lazy eval of P.Q"""
    # return ((sum(i * j for i, j in zip(r, c)) for c in zip(*Q)) for r in P)
    # TODO add shape check
    return (np.asarray([np.dot(r, Q[:, i]) for i in range(Q.shape[-1])]) for r in P)


class MatrixFactorisor:
    def __init__(self, R: MatrixMapper):
        """
        P.Q ~= R
        """

    # def loss(self, ):
