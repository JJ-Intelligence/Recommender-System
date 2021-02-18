from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import TrainDataset, TestDataset, EvaluationDataset


def read_train_csv(filename: str, test_size=0.2) -> (TrainDataset, TrainDataset):
    dataset = _read_csv_to_dataframe(
        filename,
        [("user id", np.int32), ("item id", np.int32), ("user rating", np.float16), ("timestamp", np.float32)]
    )

    x_train, x_evaluation, y_train, y_evaluation = train_test_split(
        dataset[["user id", "item id", "timestamp"]],
        dataset[["user rating"]],
        test_size=test_size,
        shuffle=True)

    return TrainDataset(x_train, y_train), EvaluationDataset(x_evaluation, y_evaluation)


def read_test_csv(filename: str) -> TestDataset:
    dataset = _read_csv_to_dataframe(
        filename,
        [("user id", np.int32), ("item id", np.int32), ("timestamp", np.float32)]
    )
    return TestDataset(dataset)


def _read_csv_to_dataframe(filename: str, columns: List[Tuple[str, np.dtype]]) -> pd.DataFrame:
    dataset = pd.read_csv(
        filename,
        names=[name for name, _ in columns],
    )

    def col_to_type(col_name, data_type):
        dataset[col_name] = pd.to_numeric(dataset[col_name], errors="coerce").dropna().astype(data_type)

    for name, col_type in columns:
        col_to_type(name, col_type)

    return dataset


def write_output_csv(filename: str, predictions):
    pass
