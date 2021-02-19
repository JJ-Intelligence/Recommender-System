import numpy as np
import pandas as pd
from pytest import fixture

from models.matrix_fact import DictMatrix
from src import TrainDataset


@fixture
def dataset():
    dataset = pd.DataFrame({
        "user id": [5, 10, 20, 24, 10, 5],
        "item id": [1, 3, 1, 2, 5, 9],
        "timestamp": [1, 2, 3, 4, 5, 6],
    })
    ratings = pd.Series([1, 4, 0.5, 4.5, 3, 2.5])
    return TrainDataset(dataset, ratings)


@fixture
def dict_matrix(dataset):
    return DictMatrix(dataset)


def test_dict_matrix_size(dict_matrix):
    assert dict_matrix.num_items() == 5
    assert dict_matrix.num_users() == 4
