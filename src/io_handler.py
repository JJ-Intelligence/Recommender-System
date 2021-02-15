import pandas
from sklearn.model_selection import train_test_split

from src.data import TrainDataset, TestDataset, EvaluationDataset


def read_train_csv(filename: str, test_size=0.2) -> (TrainDataset, TrainDataset):
    dataset = pandas.read_csv(filename)
    train_data, evaluation_data = train_test_split(dataset, test_size, shuffle=True)
    return TrainDataset(train_data), EvaluationDataset(evaluation_data)


def read_test_csv(filename: str) -> TestDataset:
    return TestDataset(pandas.read_csv(filename))


def write_output_csv(filename: str, predictions):
    pass
