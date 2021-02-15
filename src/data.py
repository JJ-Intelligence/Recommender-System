import pandas


class TrainDataSet:
    def __init__(self, dataset: pandas.DataFrame):
        self.dataset = dataset


class TestDataSet:
    def __init__(self, dataset: pandas.DataFrame):
        self.dataset = dataset
