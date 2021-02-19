from abc import ABC, abstractmethod

import numpy as np
from numpy import linalg as LA

from src import TrainDataset


class CustomMatrix(ABC):
    def __init__(self, shape):
        self.shape = shape


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

        super().__init__((self.get_width(), self.get_height()))

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

        if user_id in self.users:  # if the user doesn't exist, return row of 0s
            user_items = self.users.get(user_id)
            for item_id in user_items:
                vec[item_id] = user_items[item_id]

        return vec


class LazyMatrix(CustomMatrix):
    def __init__(self, generator, shape):
        super().__init__(shape)
        self.generator = generator

    def __call__(self, *args, **kwargs):
        return np.asarray(list(self.generator))

    def __iter__(self):
        return self.generator.__iter__()

    def __next__(self):
        return self.generator.__next__()

    @staticmethod
    def lazy_dot(P: np.ndarray, Q: np.ndarray):
        """Lazy eval of P.Q"""
        matrix = LazyMatrix(
            (np.asarray([np.dot(r, Q[:, i]) for i in range(Q.shape[-1])]) for r in P),
            (P.shape[0], Q.shape[-1])
        )
        assert matrix.shape == (P.shape[0], Q.shape[-1])
        return matrix

    @staticmethod
    def lazy_sub(R: SparseMatrix, PQ):
        """Lazy eval of R - PQ"""
        assert R.shape == PQ.shape
        matrix = LazyMatrix((np.subtract(R.get_item_vec(i + 1) - row) for i, row in enumerate(PQ)), PQ.shape)
        assert matrix.shape == PQ.shape
        return matrix

    @staticmethod
    def lazy_pow(P, power: int):
        """ Raises each value in matrix P to the given power"""
        return LazyMatrix((row ** power for row in P.generator), P.shape)

    @staticmethod
    def lazy_mean(P):
        return LazyMatrix((np.mean(row) for row in P), (P.shape[0],))


class MatrixFactoriser:
    def __init__(self, R: SparseMatrix):
        """
        P.Q ~= R
        """

    @staticmethod
    def loss_mse(R: SparseMatrix, P: np.array, Q: np.array, l: int):
        """Returns the MSE loss for matrix factorization, where R ~= P.Q"""
        return \
            np.mean(LazyMatrix.lazy_pow(LazyMatrix.lazy_sub(R, LazyMatrix.lazy_dot(P, Q)), 2), axis=0) \
            + l * (LA.norm(P) ** 2 + LA.norm(Q) ** 2)
