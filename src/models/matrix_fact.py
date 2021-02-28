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
    user_indices = users_ratings[:, 0].astype(np.int32)
    item_indices = users_ratings[:, 1].astype(np.int32)
    ratings = users_ratings[:, 2].astype(np.float32)

    for i in range(0, len(users_ratings), batch_size):
        dmse_dbu, dmse_dbi, dmse_dh, dmse_dw, user_indices_update, item_indices_update = _train_batch(
            user_indices[i:i + batch_size],
            item_indices[i:i + batch_size],
            ratings[i:i + batch_size],
            mu, bu, bi, H, W, lr, reg_bu, reg_bi, reg_H, reg_W)

        # print("BU", bu)
        # print("dmse_dbu", dmse_dbu)
        # print("BI", bi)
        # print("dmse_dbi", dmse_dbi)
        # print()
        # print("H", H)
        # print("dmse_dh", dmse_dh)
        # print("W", W)
        # print("dmse_dw", dmse_dw)

        # Update weights, using loss gradient changes
        bu[user_indices_update] += dmse_dbu
        bi[item_indices_update] += dmse_dbi
        H[user_indices_update, :] += dmse_dh
        W[:, item_indices_update] += dmse_dw


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

    # Group and reduce updates
    user_indices_reduced = list(set(user_indices))
    item_indices_reduced = list(set(item_indices))

    dmse_dbu_reduced = reduce_grad_change(dmse_dbu, user_indices, user_indices_reduced)
    dmse_dbi_reduced = reduce_grad_change(dmse_dbi, item_indices, item_indices_reduced)
    dmse_dh_reduced = reduce_grad_change(dmse_dh, user_indices, user_indices_reduced)
    dmse_dw_reduced = reduce_grad_change(dmse_dw.T, item_indices, item_indices_reduced).T
    assert len(dmse_dbu_reduced) == len(user_indices_reduced)
    assert len(dmse_dbi_reduced) == len(item_indices_reduced)
    assert len(dmse_dh_reduced) == len(user_indices_reduced)
    assert dmse_dw_reduced.shape[-1] == len(item_indices_reduced)

    return dmse_dbu_reduced, dmse_dbi_reduced, dmse_dh_reduced, dmse_dw_reduced, user_indices_reduced, \
           item_indices_reduced


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
    print("Making prediction ---------")
    print("mu + bu + bi = ", mu + bu[user_indices] + bi[item_indices])
    print("Max, min = ", np.max(mu + bu[user_indices] + bi[item_indices]),
          np.min(mu + bu[user_indices] + bi[item_indices]))
    print("H . W = ", np.sum(H[user_indices, :] * W[:, item_indices].T, axis=1))
    print("Max, min = ", np.max(np.sum(H[user_indices, :] * W[:, item_indices].T, axis=1)),
          np.min(np.sum(H[user_indices, :] * W[:, item_indices].T, axis=1)))
    pred = mu + bu[user_indices] + bi[item_indices] + np.sum(H[user_indices, :] * W[:, item_indices].T, axis=1)
    print("prediction = ", pred)
    print("Max, min = ", np.max(pred), np.min(pred))
    print("Pred shape", pred.shape)

    return mu + bu[user_indices] + bi[item_indices] + np.sum(H[user_indices, :] * W[:, item_indices].T, axis=1)


class MatrixFactoriser(ModelBase):
    def __init__(self):
        super().__init__()
        self.H = self.W = self.R = self.user_map = self.item_map = None

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
        with EpochBar('Training Step', max=epochs) as bar:
            for epoch in range(epochs):
                evaluation = self.train_step(train_dataset, eval_dataset, lr, batch_size, user_bias_reg, item_bias_reg,
                                             user_reg, item_reg)
                if eval_dataset is not None:
                    eval_history.append(evaluation)
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
