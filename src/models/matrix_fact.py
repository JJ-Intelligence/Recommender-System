from typing import Tuple, List

import numpy as np
from numba import njit, jit

from src import TrainDataset, ModelABC, TestDataset, EvaluationDataset, EpochBar, PercentageBar


class DictMatrix:

    def __init__(self, dataset: TrainDataset):
        """
        dataset: user id, item id, rating, timestamp

        user id + item id -> rating
        """

        self.user_map = self.series_to_index_map(dataset.dataset["user id"])
        self.item_map = self.series_to_index_map(dataset.dataset["item id"])
        self.users_ratings_len = len(dataset.dataset)
        self.users_ratings = list(map(
            lambda row: (self.user_map[np.int32(row[0])], self.item_map[np.int32(row[1])], row[2]),
            dataset.dataset.to_numpy()[:, [0, 1, 3]]
        ))

    def num_users(self):
        return len(self.user_map)

    def num_items(self):
        return len(self.item_map)

    def get_user_item_maps(self):
        return self.user_map, self.item_map

    @staticmethod
    def series_to_index_map(series):
        return {val: index for index, val in enumerate(series.unique())}


def train_step(users_ratings, H: np.ndarray, W: np.ndarray, lr: float):
    repeat = 10000
    with PercentageBar('Training Step', max=len(users_ratings)/repeat) as bar:
        for i, (user_index, item_index, rating) in enumerate(users_ratings):
            pred = predict_user_item_rating(H, W, user_index, item_index)
            diff = lr * 2 * (rating - pred)

            dmse_dh = diff * W[:, item_index]
            dmse_dw = diff * H[user_index, :]

            H[user_index, :] += dmse_dh
            W[:, item_index] += dmse_dw
            if i % repeat == 0:
                bar.next()


def predict_user_item_rating(H: np.ndarray, W: np.ndarray, user_index: int, item_index: int):
    return H[user_index, :].dot(W[:, item_index])


class MatrixFactoriser(ModelABC):
    def __init__(self, k: int, hw_init: float):
        self.k = k
        self.hw_init = hw_init

        self.H = self.W = None
        self.user_map = self.item_map = None

    def train(self, dataset: TrainDataset, eval_dataset: EvaluationDataset = None, epochs: int = 1, lr: float = 0.001):
        R = DictMatrix(dataset)
        self.user_map, self.item_map = R.get_user_item_maps()

        self.H = np.full((R.num_users(), self.k), self.hw_init, dtype=np.float32)
        self.W = np.full((self.k, R.num_items()), self.hw_init, dtype=np.float32)

        eval_history = []
        with EpochBar('Training Step', max=epochs) as bar:
            for epoch in range(epochs):
                print(f"\nEpoch: {epoch} / {epochs}")
                train_step(R.users_ratings, self.H, self.W, lr)
                if eval_dataset is not None:
                    print("Evaluation:")
                    eval_result = self.eval(eval_dataset)
                    eval_history.append(eval_result)
                    print(eval_result)
                bar.next()

        return eval_history

    def predict(self, dataset: TestDataset) -> np.ndarray:

        def _pred(user_id, item_id):
            if user_id in self.user_map:
                user_index = self.user_map[user_id]
                if item_id in self.item_map:
                    item_index = self.item_map[item_id]
                    return predict_user_item_rating(self.H, self.W, user_index, item_index)
                else:
                    # TODO add some case for a new item (cold start)
                    return 3.0
            else:
                # TODO add some case for a new user (cold start)
                return 3.0

        return np.asarray(
            [_pred(user_id, item_id) for user_id, item_id in dataset.dataset[["user id", "item id"]].to_numpy()],
            dtype=np.float16
        )

        # predictions = []
        # for i, (user_id, item_id, timestamp) in dataset.dataset.iterrows():
        #     if user_id in self.user_map:
        #         user_index = self.user_map[user_id]
        #         if item_id in self.item_map:
        #             item_index = self.item_map[item_id]
        #             pred = predict_user_item_rating(self.H, self.W, user_index, item_index)
        #             predictions.append(pred)
        #         else:
        #             # TODO add some case for a new item (cold start)
        #             predictions.append(0)
        #     else:
        #         # TODO add some case for a new user (cold start)
        #         predictions.append(0)

        # return np.asarray(predictions, dtype=np.float16)
