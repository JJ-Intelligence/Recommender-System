import numpy as np
import pandas as pd
from pytest import fixture

from data import EvaluationDataset, TrainDataset
from models.matrix_fact import DictMatrix, MatrixFactoriser, _train_batch, _predict_ratings


@fixture
def train_dataset():
    dataset = pd.DataFrame({
        "user id": [5, 10, 20, 24, 10, 5],
        "item id": [1, 3, 1, 2, 5, 9],
        "timestamp": [1, 2, 3, 4, 5, 6],
    })
    ratings = pd.Series([1, 4, 0.5, 4.5, 3, 2.5], dtype=np.float16)
    return TrainDataset(dataset, ratings)


@fixture
def eval_dataset():
    dataset = pd.DataFrame({
        "user id": [10, 2, 24, 5, 20],
        "item id": [9, 2, 5, 1, 4],
        "timestamp": [1, 2, 3, 4, 5],
    })
    ratings = pd.Series([1, 4, 0.5, 4.5, 3], dtype=np.float16)
    return EvaluationDataset(dataset, ratings)


@fixture
def dict_matrix(train_dataset):
    return DictMatrix(train_dataset)


def test_dict_matrix_size(dict_matrix):
    assert dict_matrix.num_items() == 5
    assert dict_matrix.num_users() == 4


# def test_matrix_factoriser_train(train_dataset, eval_dataset):
#     factoriser = MatrixFactoriser(k=10, hw_init=0.1)
#     factoriser.train(train_dataset, eval_dataset=eval_dataset)


def test_predict_ratings():
    k = 4
    mu = 3.4
    bu = np.random.normal(0, 0.1, size=(9,))
    bi = np.random.normal(0, 0.1, size=(20,))
    H = np.random.normal(0, 1, size=(9, k))
    W = np.random.normal(0, 2, size=(k, 20))
    user_indices = np.asarray([0, 1, 2, 1, 1, 0, 7, 4, 3])
    item_indices = np.asarray([0, 5, 10, 19, 5, 0, 12, 12, 13])

    # predict_ratings result
    result = _predict_ratings(mu, bu, bi, H, W, user_indices, item_indices)

    # Manually make a prediction
    expected = []
    for (user_index, item_index) in zip(user_indices, item_indices):
        expected.append(mu + bu[user_index] + bi[item_index] + H[user_index, :].dot(W[:, item_index]))
    expected = np.asarray(expected)

    assert result.shape == (9,)
    assert np.allclose(expected, result)
