import pandas
from sklearn.model_selection import train_test_split

from src.data import TrainDataset, TestDataset, EvaluationDataset


def read_train_csv(filename: str, test_size=0.2) -> (TrainDataset, TrainDataset):
    dataset = pandas.read_csv(filename, names=["user id", "item id", "user rating", "timestamp"])

    x_train, x_evaluation, y_train, y_evaluation = train_test_split(
        dataset[["user id", "item id", "timestamp"]],
        dataset[["user rating"]],
        test_size=test_size,
        shuffle=True)

    return TrainDataset(x_train, y_train), EvaluationDataset(x_evaluation, y_evaluation)


def read_test_csv(filename: str) -> TestDataset:
    return TestDataset(pandas.read_csv(filename))


def write_output_csv(filename: str, predictions):
    pass
