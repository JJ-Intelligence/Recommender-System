import numpy as np
from numba import njit

from src import TrainDataset, ModelABC, TestDataset, EvaluationDataset, EpochBar


class DictMatrix:

    def __init__(self, dataset: TrainDataset):
        """
        dataset: user id, item id, rating, timestamp
        """

        self.user_map = self.series_to_index_map(dataset.dataset["user id"])
        self.item_map = self.series_to_index_map(dataset.dataset["item id"])
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
def _train_step(users_ratings: np.ndarray, H: np.ndarray, W: np.ndarray, batch_size: int, lr: float):
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
                 lr: float, alpha: float = 0, beta: float = 0):
    predictions = _predict_ratings(H, W, user_indices, item_indices)
    residuals = 2 * (ratings - predictions)
    dmse_dh = lr * ((residuals * W[:, item_indices]).T + (alpha * H[user_indices, :]))
    dmse_dw = lr * ((residuals * H[user_indices, :].T) + (beta * W[:, item_indices]))

    return dmse_dh, dmse_dw


@njit(parallel=True)
def _predict_ratings(H: np.ndarray, W: np.ndarray, user_indices: np.ndarray, item_indices: np.ndarray):
    # Perform a point-wise dot product
    return np.sum(H[user_indices, :] * W[:, item_indices].T, axis=1)


class MatrixFactoriser(ModelABC):
    def __init__(self, k: int, hw_init: float):
        self.k = k
        self.hw_init = hw_init

        self.H = self.W = None
        self.user_map = self.item_map = None

    def train(self, dataset: TrainDataset, eval_dataset: EvaluationDataset = None, epochs: int = 10, lr: float = 0.001):
        R = DictMatrix(dataset)
        self.user_map, self.item_map = R.get_user_item_maps()

        norm_mean = 0
        norm_stddev = 0.5
        self.H = np.random.normal(norm_mean, norm_stddev, (R.num_users(), self.k)).astype(np.float32)
        self.W = np.random.normal(norm_mean, norm_stddev, (self.k, R.num_items())).astype(np.float32)

        eval_history = []
        # Training epochs
        with EpochBar('Training Step', max=epochs) as bar:
            for epoch in range(epochs):
                _train_step(R.users_ratings, self.H, self.W, batch_size=5_000, lr=lr)

                # Evaluate at the end of the epoch
                if eval_dataset is not None:
                    eval_result = self.eval(eval_dataset)
                    eval_history.append(eval_result)
                    bar.mse = eval_result.mse
                bar.next()

        return eval_history

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
