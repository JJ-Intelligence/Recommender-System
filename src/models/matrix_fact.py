import numpy as np
from numba import njit

from data import TrainDataset, EvaluationDataset, TestDataset
from models.model_base import ModelBase
from progress_bars import EpochBar


class DictMatrix:

    def __init__(self, dataset: TrainDataset, maps=None):
        """
        dataset: user id, item id, rating, timestamp
        """
        if maps is None:
            self.user_map = self.series_to_index_map(dataset.dataset["user id"])
            self.item_map = self.series_to_index_map(dataset.dataset["item id"])
        else:
            self.user_map, self.item_map = maps

        self.users_ratings = np.asarray([
            [self.user_map[np.int32(row[0])], self.item_map[np.int32(row[1])], row[2]]
            for row in dataset.dataset.to_numpy()[:, [0, 1, 3]]
        ])

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
def do_train_step(users_ratings: np.ndarray,
                  mu: np.float32,
                  bu: np.ndarray,
                  bi: np.ndarray,
                  H: np.ndarray,
                  W: np.ndarray,
                  batch_size: int,
                  lr: float,
                  reg_bu: float,
                  reg_bi: float,
                  reg_H: float,
                  reg_W: float):
    """ Perform a single training step (1 epoch) """
    np.random.shuffle(users_ratings)

    user_indices = users_ratings[:, 0].astype(np.int32)
    item_indices = users_ratings[:, 1].astype(np.int32)
    ratings = users_ratings[:, 2].astype(np.float32)

    for i in range(0, len(users_ratings), batch_size):
        dmse_dbu, dmse_dbi, dmse_dh, dmse_dw = _train_batch(
            user_indices[i:i + batch_size],
            item_indices[i:i + batch_size],
            ratings[i:i + batch_size],
            mu, bu, bi, H, W, lr, reg_bu, reg_bi, reg_H, reg_W
        )

        # Update weights, using loss gradient changes
        bu[user_indices[i:i + batch_size]] += dmse_dbu
        bi[item_indices[i:i + batch_size]] += dmse_dbi
        H[user_indices[i:i + batch_size], :] += dmse_dh
        W[:, item_indices[i:i + batch_size]] += dmse_dw


@njit(parallel=True)
def _train_batch(user_indices: np.ndarray,
                 item_indices: np.ndarray,
                 ratings: np.ndarray,
                 mu: np.float32,
                 bu: np.ndarray,
                 bi: np.ndarray,
                 H: np.ndarray,
                 W: np.ndarray,
                 lr: float,
                 reg_bu: float,
                 reg_bi: float,
                 reg_H: float,
                 reg_W: float):
    predictions = _predict_ratings(mu, bu, bi, H, W, user_indices, item_indices)
    residuals = 2 * (ratings - predictions)

    # Gradient changes
    dmse_dbu = lr * (residuals - (reg_bu * bu[user_indices]))
    dmse_dbi = lr * (residuals - (reg_bi * bi[item_indices]))
    dmse_dh = lr * ((residuals * W[:, item_indices]).T - (reg_H * H[user_indices, :]))
    dmse_dw = lr * ((residuals * H[user_indices, :].T) - (reg_W * W[:, item_indices]))

    return dmse_dbu, dmse_dbi, dmse_dh, dmse_dw


@njit(parallel=True)
def reduce_grad_change(grad_change, indices, indices_reduced):
    return np.asarray([grad_change[indices == i].sum(axis=0) for i in indices_reduced])


@njit(parallel=True)
def _predict_ratings(mu: np.float32,
                     bu: np.ndarray,
                     bi: np.ndarray,
                     H: np.ndarray,
                     W: np.ndarray,
                     user_indices: np.ndarray,
                     item_indices: np.ndarray):
    # Perform a point-wise dot product
    return mu + bu[user_indices] + bi[item_indices] + np.sum(H[user_indices, :] * W[:, item_indices].T, axis=1)


class MatrixFactoriser(ModelBase):
    def __init__(self):
        super().__init__()
        self.H = self.W = self.R = self.user_map = self.item_map = self.mu = self.bu = self.bi = \
            self.k = self.hw_init_stddev = None

    def initialise(self, k: int, hw_init_stddev: float):
        self.k = k
        self.hw_init_stddev = hw_init_stddev

    def setup_model(self, train_dataset: TrainDataset):

        if self.item_map is not None and self.user_map is not None:
            # maps have been preloaded from file
            self.R = DictMatrix(train_dataset, (self.user_map, self.item_map))
        else:
            # maps not yet loaded
            self.R = DictMatrix(train_dataset)
        self.user_map, self.item_map = self.R.get_user_item_maps()

        # Latent feature matrices
        norm_mean = 0
        norm_stddev = self.hw_init_stddev
        self.H = np.random.normal(norm_mean, norm_stddev, (self.R.num_users(), self.k)).astype(np.float32)
        self.W = np.random.normal(norm_mean, norm_stddev, (self.k, self.R.num_items())).astype(np.float32)

        # Biases (global mean, user bias, item bias)
        self.mu = np.mean(self.R.users_ratings[:, -1], dtype=np.float32)
        self.bu = np.zeros(self.R.num_users(), dtype=np.float32)
        self.bi = np.zeros(self.R.num_items(), dtype=np.float32)

        # Add 0 rows for unknown items or users
        self.H = np.append(self.H, np.full((1, self.k), 0), axis=0)
        self.W = np.append(self.W, np.full((self.k, 1), 0), axis=1)
        self.bu = np.append(self.bu, 0)
        self.bi = np.append(self.bi, 0)

    def train(self,
              train_dataset: TrainDataset,
              eval_dataset: EvaluationDataset = None,
              epochs: int = 10,
              lr: float = 0.01,
              batch_size: int = 100_000,
              user_bias_reg: float = 0.01,
              item_bias_reg: float = 0.01,
              user_reg: float = 0.01,
              item_reg: float = 0.01):
        """For debug mode - call train step when using trainer"""

        self.setup_model(train_dataset)
        eval_history = []
        # Training epochs
        with EpochBar('Training Step', max=epochs, check_tty=False) as bar:
            for epoch in range(epochs):
                print("Starting epoch", epoch)
                evaluation = self.train_step(train_dataset, eval_dataset, lr, batch_size, user_bias_reg, item_bias_reg,
                                             user_reg, item_reg)
                if eval_dataset is not None:
                    print("MSE: ", evaluation.mse)
                    eval_history.append(evaluation)

                    if evaluation.mse > bar.mse:
                        lr /= 2
                        print("Decreasing lr:", lr)
                    bar.mse = evaluation.mse

                bar.next()

        return eval_history

    def train_step(self,
                   train_dataset: TrainDataset,
                   eval_dataset: EvaluationDataset,
                   lr: float = 0.01,
                   batch_size: int = 100_000,
                   user_bias_reg: float = 0.01,
                   item_bias_reg: float = 0.01,
                   user_reg: float = 0.01,
                   item_reg: float = 0.01):

        if self.R is None:
            self.setup_model(train_dataset)

        do_train_step(
            self.R.users_ratings,
            self.mu,
            self.bu,
            self.bi,
            self.H,
            self.W,
            batch_size,
            lr,
            user_bias_reg,
            item_bias_reg,
            user_reg,
            item_reg
        )

        # Evaluate at the end of the epoch
        if eval_dataset is not None:
            return self.eval(eval_dataset)

    def predict(self, dataset: TestDataset) -> np.ndarray:

        # Convert user/item ids into indices
        def id_to_index(_id, index_map):
            if _id in index_map:
                return index_map[_id]

            return len(index_map)

        user_item_data = np.asarray([
            [id_to_index(np.int32(row[0]), self.R.user_map), id_to_index(np.int32(row[1]), self.R.item_map)]
            for row in dataset.dataset[["user id", "item id"]].to_numpy()
        ])
        user_indices = user_item_data[:, 0]
        item_indices = user_item_data[:, 1]

        return _predict_ratings(self.mu, self.bu, self.bi, self.H, self.W, user_indices, item_indices)

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
