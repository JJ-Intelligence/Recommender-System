from typing import Tuple, List, Union

import tensorflow as tf
import numpy as np
from numba import njit, jit

from data import TrainDataset, EvaluationDataset, TestDataset
from evaluation import Evaluation
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

        user_indices = []
        item_indices = []
        ratings = []
        for row in dataset.dataset.to_numpy()[:, [0, 1, 3]]:
            user_indices.append(self.user_map[tf.int32(row[0])])
            item_indices.append(self.item_map[tf.int32(row[1])])
            ratings.append(self.user_map[tf.float32(row[2])])

        self.user_indices = tf.convert_to_tensor(user_indices)
        self.item_indices = tf.convert_to_tensor(item_indices)
        self.ratings = tf.convert_to_tensor(ratings)

    def num_users(self) -> int:
        return len(self.user_map)

    def num_items(self) -> int:
        return len(self.item_map)

    def get_user_item_maps(self) -> Tuple[dict, dict]:
        return self.user_map, self.item_map

    def get_ratings_tensors(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return self.user_indices, self.item_indices, self.ratings

    @staticmethod
    def series_to_index_map(series) -> dict:
        return {val: index for index, val in enumerate(series.unique())}


def do_train_step(user_indices: tf.Tensor,
                  item_indices: tf.Tensor,
                  ratings: tf.Tensor,
                  user_indices_inv_counts: tf.Tensor,
                  item_indices_inv_counts: tf.Tensor,
                  mu: tf.float32,
                  bu: tf.Tensor,
                  bi: tf.Tensor,
                  H: tf.Tensor,
                  W: tf.Tensor,
                  batch_size: int,
                  lr: float,
                  reg_bu: float,
                  reg_bi: float,
                  reg_H: float,
                  reg_W: float):
    """ Perform a single training step (1 epoch) """
    for i in range(0, len(ratings), batch_size):
        user_indices_batch = user_indices[i:i + batch_size]
        item_indices_batch = item_indices[i:i + batch_size]

        dmse_dbu, dmse_dbi, dmse_dh, dmse_dw = _train_batch(
            user_indices_batch,
            item_indices_batch,
            ratings[i:i + batch_size],
            user_indices_inv_counts[i:i + batch_size],
            item_indices_inv_counts[i:i + batch_size],
            mu, bu, bi, H, W, lr, reg_bu, reg_bi, reg_H, reg_W)

        # Update weights, using loss gradient changes
        bu.scatter_nd_add(tf.reshape(user_indices_batch, (-1, 1)), dmse_dbu)
        bi.scatter_nd_add(tf.reshape(item_indices_batch, (-1, 1)), dmse_dbi)
        H.scatter_nd_add(tf.reshape(user_indices_batch, (-1, 1)), dmse_dh)

        # To update W, we need to update each individual value in it, as it's of shape (k, num_items), but dmse_dw is
        # of shape (num_items, k)
        feature_indices = tf.reshape(tf.convert_to_tensor(list(range(W.shape[0]))), (-1, 1))
        W_update_item_indices = [tf.concat([feature_indices, tf.fill((len(feature_indices), 1), item_index)], axis=1)
                                 for item_index in item_indices_batch]
        W.scatter_nd_add(W_update_item_indices, dmse_dw)


def _train_batch(user_indices: tf.Tensor,
                 item_indices: tf.Tensor,
                 ratings: tf.Tensor,
                 user_inv_counts: tf.Tensor,
                 item_inv_counts: tf.Tensor,
                 mu: tf.float32,
                 bu: tf.Tensor,
                 bi: tf.Tensor,
                 H: tf.Tensor,
                 W: tf.Tensor,
                 lr: float,
                 reg_bu: float,
                 reg_bi: float,
                 reg_H: float,
                 reg_W: float) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    predictions = _predict_ratings(mu, bu, bi, H, W, user_indices, item_indices)
    residuals = ratings - predictions

    # Gradient changes
    dmse_dbu = lr * ((residuals * user_inv_counts) - (reg_bu * tf.gather(bu, user_indices)))
    dmse_dbi = lr * ((residuals * item_inv_counts) - (reg_bi * tf.gather(bi, item_indices)))

    dmse_dh = lr * (tf.transpose(user_inv_counts * residuals * tf.gather(W, item_indices, axis=0)) -
                    (reg_H * tf.gather(H, user_indices, axis=1)))
    dmse_dw = lr * ((item_inv_counts * residuals * tf.transpose(tf.gather(H, user_indices, axis=1))) -
                    (reg_W * tf.gather(W, item_indices, axis=0)))
    return dmse_dbu, dmse_dbi, dmse_dh, dmse_dw


def _predict_ratings(mu: tf.float32,
                     bu: tf.Tensor,
                     bi: tf.Tensor,
                     H: tf.Tensor,
                     W: tf.Tensor,
                     user_indices: tf.Tensor,
                     item_indices: tf.Tensor) -> tf.Tensor:
    # Perform a point-wise matrix dot product
    return mu + tf.gather(bu, user_indices) + tf.gather(bi, item_indices) + \
           tf.reduce_sum(tf.gather(H, user_indices, axis=0) * tf.transpose(tf.gather(W, item_indices, axis=1)), axis=1)


class MatrixFactoriser(ModelBase):
    def __init__(self):
        super().__init__()
        self.k = self.hw_init_stddev = self.mu = self.bu = self.bi = self.H = self.W = self.R = self.user_map = \
            self.item_map = None

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
        self.H = tf.random.normal((self.R.num_users(), self.k), norm_mean, norm_stddev, dtype=tf.dtypes.float32)
        self.W = tf.random.normal((self.k, self.R.num_items()), norm_mean, norm_stddev, dtype=tf.dtypes.float32)

        # Biases (global mean, user bias, item bias)
        self.mu = tf.reduce_mean(self.R.get_ratings_tensors()[-1], dtype=tf.dtypes.float32)
        self.bu = tf.zeros(self.R.num_users() + 1, dtype=tf.dtypes.float32)  # Extra 0 row for unknown items/users
        self.bi = tf.zeros(self.R.num_items() + 1, dtype=tf.dtypes.float32)

        # Add 0 rows for unknown items or users
        self.H = tf.concat([self.H, tf.fill((1, self.k), 0)], axis=0)
        self.W = tf.concat([self.W, tf.fill((self.k, 1), 0)], axis=1)

    def train(self,
              train_dataset: TrainDataset,
              eval_dataset: EvaluationDataset = None,
              epochs: int = 10,
              lr: float = 0.01,
              batch_size: int = 100_000,
              user_bias_reg: float = 0.01,
              item_bias_reg: float = 0.01,
              user_reg: float = 0.01,
              item_reg: float = 0.01) -> List[Evaluation]:
        """For debug mode - call train step when using trainer"""

        self.setup_model(train_dataset)
        eval_history = []

        # Training epochs
        with EpochBar('Training Step', max=epochs) as bar:
            for epoch in range(epochs):
                evaluation = self.train_step(
                    train_dataset, eval_dataset, lr, batch_size, user_bias_reg, item_bias_reg, user_reg, item_reg)
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
                   item_reg: float = 0.01) -> Union[None, Evaluation]:

        if self.R is None:
            self.setup_model(train_dataset)

        user_indices, item_indices, ratings = self.R.get_ratings_tensors()
        user_indices_inv_counts = self.indices_to_inv_counts(user_indices)
        item_indices_inv_counts = self.indices_to_inv_counts(item_indices)

        do_train_step(
            user_indices, item_indices, ratings, user_indices_inv_counts, item_indices_inv_counts, self.mu, self.bu,
            self.bi, self.H, self.W, batch_size, lr, user_bias_reg, item_bias_reg, user_reg, item_reg,
        )

        # Evaluate at the end of the epoch
        if eval_dataset is not None:
            return self.eval(eval_dataset)

    @staticmethod
    def indices_to_inv_counts(indices):
        # Map indices to an array of 1/<index_count>
        _, inv, counts = np.unique(indices, return_inverse=True, return_counts=True)
        return tf.convert_to_tensor([1 / counts[i] for i in inv])

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
