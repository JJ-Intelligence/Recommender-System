import pandas


class TrainDataset:
    """
    Data from the train file, which we want to learn from
    Split into train/evaluation datasets
    """
    def __init__(self, X: pandas.array, Y: pandas.array):
        self.X = X
        self.Y = Y


class EvaluationDataset(TrainDataset):
    pass


class TestDataset:
    """Data from the test file, which we want to generate predictions of"""
    def __init__(self, dataset: pandas.DataFrame):
        self.dataset = dataset
