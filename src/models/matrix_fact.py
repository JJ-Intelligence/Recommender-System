from typing import Tuple, List

import numpy as np
from numba import njit

from src import TrainDataset, ModelBase, TestDataset, EvaluationDataset, EpochBar, PercentageBar


class DictMatrix:

    def __init__(self, dataset: TrainDataset, maps=None):
        """
        dataset: user id, item id, rating, timestamp

        user id + item id -> rating
        """
        if maps is None:
            self.user_map = self.series_to_index_map(dataset.dataset["user id"])
            self.item_map = self.series_to_index_map(dataset.dataset["item id"])
        else:
            self.user_map, self.item_map = maps

        self.users_ratings = np.asarray(list(map(
            lambda row: [self.user_map[np.int32(row[0])], self.item_map[np.int32(row[1])], row[2]],
            dataset.dataset.to_numpy()[:, [0, 1, 3]]
        )))

    def num_users(self):
        return len(self.user_map)

    def num_items(self):
        return len(self.item_map)

    def get_user_item_maps(self):
        return self.user_map, self.item_map

    @staticmethod
    def series_to_index_map(series):
        return {val: index for index, val in enumerate(series.unique())}


@njit
def do_train_step(users_ratings: np.ndarray, H: np.ndarray, W: np.ndarray, batch_size: int, lr: float):
    """ Perform a single training step (1 epoch) """
    user_indices = users_ratings[:, 0].astype(np.int32)
    item_indices = users_ratings[:, 1].astype(np.int32)
    ratings = users_ratings[:, 2].astype(np.float32)

    for i in range(0, len(users_ratings), batch_size):
        dmse_dh, dmse_dw = _train_batch(
            user_indices[i:i + batch_size],
            item_indices[i:i + batch_size],
            ratings[i:i + batch_size],
            H, W, lr)

        H[user_indices[i:i + batch_size], :] += dmse_dh
        W[:, item_indices[i:i + batch_size]] += dmse_dw


@njit(parallel=True)
def _train_batch(user_indices: np.ndarray, item_indices: np.ndarray, ratings: np.ndarray, H: np.ndarray, W: np.ndarray,
                 lr: float):
    predictions = _predict_ratings(H, W, user_indices, item_indices)
    diffs = lr * 2 * (ratings - predictions)

    dmse_dh = (diffs * W[:, item_indices]).T
    dmse_dw = diffs * H[user_indices, :].T

    return dmse_dh, dmse_dw


@njit(parallel=True)
def _predict_ratings(H: np.ndarray, W: np.ndarray, user_indices: np.ndarray, item_indices: np.ndarray):
    # Perform a point-wise dot product
    return np.sum(H[user_indices, :] * W[:, item_indices].T, axis=1)


class MatrixFactoriser(ModelBase):
    def __init__(self):
        super().__init__()
        self.H = self.W = self.R = self.user_map = self.item_map = None

    def initialise(self, k: int, hw_init: float):
        self.k = k
        self.hw_init = hw_init

    def setup_model(self, train_dataset: TrainDataset):

        if self.item_map is not None and self.user_map is not None:
            # maps have been preloaded from file
            self.R = DictMatrix(train_dataset, (self.user_map, self.item_map))
        else:
            # maps not yet loaded
            self.R = DictMatrix(train_dataset)
        self.user_map, self.item_map = self.R.get_user_item_maps()

        norm_mean = 0
        norm_stddev = 0.5
        self.H = np.random.normal(norm_mean, norm_stddev, (self.R.num_users(), self.k)).astype(np.float32)
        self.W = np.random.normal(norm_mean, norm_stddev, (self.k, self.R.num_items())).astype(np.float32)

    def train(self, train_dataset: TrainDataset,
              eval_dataset: EvaluationDataset = None,
              epochs: int = 10,
              lr: float = 0.001,
              batch_size=100_000):
        """For debug mode - call train step when using trainer"""

        self.setup_model(train_dataset)
        eval_history = []
        # Training epochs
        with EpochBar('Training Step', max=epochs) as bar:
            for epoch in range(epochs):
                do_train_step(self.R.users_ratings, self.H, self.W, batch_size=batch_size, lr=lr)

                # Evaluate at the end of the epoch
                if eval_dataset is not None:
                    eval_result = self.eval(eval_dataset)
                    eval_history.append(eval_result)
                    bar.mse = eval_result.mse
                bar.next()

        return eval_history

    def train_step(self, train_dataset: TrainDataset, eval_dataset: EvaluationDataset, lr: float = 0.001, batch_size=100_000):

        if self.R is None:
            self.setup_model(train_dataset)

        do_train_step(self.R.users_ratings, self.H, self.W, batch_size=batch_size, lr=lr)

        # Evaluate at the end of the epoch
        if eval_dataset is not None:
            return self.eval(eval_dataset)

    def predict(self, dataset: TestDataset) -> np.ndarray:
        def _pred(user_id, item_id):
            if user_id in self.user_map:
                user_index = self.user_map[user_id]
                if item_id in self.item_map:
                    item_index = self.item_map[item_id]
                    return self.H[user_index, :].dot(self.W[:, item_index])
                else:
                    # TODO add some case for a new item (cold start)
                    return 3.0
            else:
                # TODO add some case for a new user (cold start)
                return 3.0

        return np.asarray(
            [_pred(user_id, item_id) for user_id, item_id in dataset.dataset[["user id", "item id"]].to_numpy()],
            dtype=np.float32
        )

    def save(self, checkpoint_file):
        user_map_np = np.array(list(self.user_map.items()), dtype="i4,i4")
        item_map_np = np.array(list(self.item_map.items()), dtype="i4,i4")
        np.savez(checkpoint_file, H=self.H, W=self.W, user_map=user_map_np, item_map=item_map_np)

    def load(self, checkpoint_file):
        npzfile = np.load(checkpoint_file)
        self.H = npzfile["H"]
        self.W = npzfile["W"]
        self.user_map, self.item_map = extra_dict_from_np(npzfile["user_map"]), extra_dict_from_np(npzfile["item_map"])


def extra_dict_from_np(np_arr: np.ndarray):
    return {key: value for key, value in np_arr}
