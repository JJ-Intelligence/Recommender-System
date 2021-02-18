from abc import ABC

import numpy as np

from src import TrainDataset


class CustomMatrix(ABC):
    pass


class SparseMatrix(CustomMatrix):
    def __init__(self, dataset: TrainDataset):
        """
        dataset: user id, item id, rating, timestamp

        user id + item id -> rating
        """
        self.users = {}
        self.max_item = 0
        self.max_user = 0

        for i, (user_id, item_id, timestamp, rating) in dataset.dataset.iterrows():
            self.max_item = max(self.max_item, item_id)
            self.max_user = max(self.max_user, user_id)
            if user_id not in self.users:
                self.users[user_id] = {item_id: rating}
            else:
                self.users[user_id][item_id] = rating

        self.shape = (self.get_width(), self.get_height())

    def get_height(self):
        """The height matters rather than the number of users, since we're not mapping users to indices"""
        return self.max_user

    def get_width(self):
        """The width matters rather than the number of items, since we're not mapping items to indices"""
        return self.max_item

    def get(self, user_id, item_id):
        if user_id in self.users:
            return self.users.get(user_id).get(item_id, 0)
        return 0

    def get_item_vec(self, user_id):
        vec = np.zeros(self.max_item)

        if user_id in self.users: # if the user doesn't exist, return row of 0s
            user_items = self.users.get(user_id)
            for item_id in user_items:
                vec[item_id] = user_items[item_id]

        return vec


class LazyMatrix(CustomMatrix):
    def __init__(self, generator, w, h):
        self.generator = generator
        self.shape = (w, h)

    @staticmethod
    def lazy_dot(P, Q):
        """Lazy eval of P.Q"""
        # return ((sum(i * j for i, j in zip(r, c)) for c in zip(*Q)) for r in P)
        # TODO add shape check
        return LazyMatrix((np.asarray([np.dot(r, Q[:, i]) for i in range(Q.shape[-1])]) for r in P),
                          P.shape[0], Q.shape[-1])

    @staticmethod
    def lazy_sub(R: SparseMatrix, PQ):
        """
        Lazy eval of R - PQ
        :param R: matrix wrapper
        :param PQ: generator
        """
        assert R.shape == PQ.shape
        return LazyMatrix((np.subtract(R.get_item_vec() - row) for y, row in enumerate(PQ)), *PQ.shape)


class MatrixFactorisor:
    def __init__(self, R: SparseMatrix):
        """
        P.Q ~= R
        """

    @staticmethod
    def loss(R, P, Q):
        """"""
        total = 0
        # for
