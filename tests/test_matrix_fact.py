import numpy as np

from models.matrix_fact import lazy_dot


def test_lazy_dot_same_size():
    A = np.array([[1, 2],
                  [-2, 3]])
    B = np.array([[3, 4],
                  [0, 1]])
    assert np.array_equal(A.dot(B), np.array(list(lazy_dot(A, B))))


def test_lazy_dot_different_sizes():
    A = np.array([[1, 2],
                  [-2, 3],
                  [9, -1]])
    B = np.array([[3, 4.3, 7],
                  [0, 1, 1.2]])
    assert np.array_equal(A.dot(B), np.array(list(lazy_dot(A, B))))


def test_lazy_dot_large():
    A = np.ones(100).reshape(-1, 1)
    B = np.ones(100).reshape(-1, 1).T
    C = lazy_dot(A, B)
    assert np.array_equal(A.dot(B), np.array(list(C)))


def test_lazy_dot_larger():
    A = np.ones(1000000).reshape(-1, 1)
    B = np.ones(1000000).reshape(-1, 1).T
    C = lazy_dot(A, B)
    assert next(C)[3] == 1
    assert next(C)[4] == 1


def test_lazy_dot_even_larger():
    k = 200
    A = np.ones(600000).reshape(-1, k)
    B = np.ones(90000).reshape(-1, k).T
    C = lazy_dot(A, B)
    assert next(C)[3] == k
    assert next(C)[4] == k
