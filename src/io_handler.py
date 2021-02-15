import pandas

from src.data import TrainDataSet, TestDataSet


def read_train_csv(filename: str) -> TrainDataSet:
    return TrainDataSet(pandas.read_csv(filename))


def read_test_csv(filename: str) -> TestDataSet:
    return TestDataSet(pandas.read_csv(filename))


def write_output_csv(filename: str, predictions):
    pass
