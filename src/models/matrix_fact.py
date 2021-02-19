import numpy as np
from numba import njit

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
        self.users_ratings = map(
            lambda row: (self.user_map[np.int32(row[0])], self.item_map[np.int32(row[1])], row[2]),
            dataset.dataset.to_numpy()[:, [0, 1, 3]]
        )

    def num_users(self):
        return len(self.user_map)

    def num_items(self):
        return len(self.item_map)

    def get_user_item_maps(self):
        return self.user_map, self.item_map

    @staticmethod
    def series_to_index_map(series):
        return {val: index for index, val in enumerate(series.unique())}


class MatrixFactoriser(ModelABC):
    def __init__(self, k: int, hw_init: float):
        self.k = k
        self.hw_init = hw_init

        self.H = self.W = None
        self.user_map = self.item_map = None

    def train(self, dataset: TrainDataset, eval_dataset: EvaluationDataset = None, epochs: int = 1, lr: float = 0.001):
        R = DictMatrix(dataset)
        self.user_map, self.item_map = R.get_user_item_maps()

        self.H = np.full((R.num_users(), self.k), self.hw_init)
        self.W = np.full((self.k, R.num_items()), self.hw_init)

        eval_history = []
        with EpochBar('Training', max=epochs) as bar:
            for epoch in range(epochs):
                print("Epoch:", epoch)
                self.train_step(R, self.H, self.W, lr)
                if eval_dataset is not None:
                    print("Evaluation:")
                    eval_result = self.eval(eval_dataset)
                    eval_history.append(eval_result)
                    print(eval_result)
                bar.next()

        return eval_history

    # @njit(parallel=True)
    @staticmethod
    def train_step(R: DictMatrix, H: np.ndarray, W: np.ndarray, lr: float):
        with EpochBar('Training Step', max=R.users_ratings_len) as bar:
            for i, (user_index, item_index, rating) in enumerate(R.users_ratings):
                pred = MatrixFactoriser._predict_user_item_rating(H, W, user_index, item_index)
                diff = lr * 2 * (rating - pred)

                dmse_dh = diff * W[:, item_index]
                dmse_dw = diff * H[user_index, :]

                H[user_index, :] += dmse_dh
                W[:, item_index] += dmse_dw
                bar.next()

    @staticmethod
    def _predict_user_item_rating(H: np.ndarray, W: np.ndarray, user_index: int, item_index: int):
        return H[user_index, :].dot(W[:, item_index])

    def predict(self, dataset: TestDataset) -> np.ndarray:
        predictions = []

        for i, (user_id, item_id, timestamp) in dataset.dataset.iterrows():
            if user_id in self.user_map:
                user_index = self.user_map[user_id]
                if item_id in self.item_map:
                    item_index = self.item_map[item_id]
                    pred = MatrixFactoriser._predict_user_item_rating(self.H, self.W, user_index, item_index)
                    predictions.append(pred)
                else:
                    # TODO add some case for a new item (cold start)
                    predictions.append(0)
            else:
                # TODO add some case for a new user (cold start)
                predictions.append(0)

        return np.asarray(predictions, dtype=np.float16)
