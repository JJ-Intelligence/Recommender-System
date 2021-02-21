import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import TrainDataset, TestDataset, EvaluationDataset

DEFAULT_RESULTS_FOLDER = "../results/"


def read_train_csv(filename: str, test_size=0.1, eval_size=0.1) -> (TrainDataset, EvaluationDataset, EvaluationDataset):
    dataset = _read_csv_to_dataframe(
        filename,
        [("user id", np.int32), ("item id", np.int32), ("user rating", np.float32), ("timestamp", np.float32)]
    )

    x_train, x_evaluation, y_train, y_evaluation = train_test_split(
        dataset[["user id", "item id", "timestamp"]],
        dataset[["user rating"]],
        test_size=eval_size,
        shuffle=True)

    x_train, x_test, y_train, y_test = train_test_split(
        x_train,
        y_train,
        test_size=test_size/(1-eval_size),  # account for smaller train set
        shuffle=False)

    return \
        TrainDataset(x_train, y_train), \
        EvaluationDataset(x_evaluation, y_evaluation), \
        EvaluationDataset(x_test, y_test)


def read_test_csv(filename: str) -> TestDataset:
    dataset = _read_csv_to_dataframe(
        filename,
        [("user id", np.int32), ("item id", np.int32), ("timestamp", np.float32)]
    )
    return TestDataset(dataset)


def _read_csv_to_dataframe(filename: str, columns) -> pd.DataFrame:
    dataset = pd.read_csv(
        filename,
        names=[name for name, _ in columns],
    )

    # Set all non-numeric values to NaN
    for name, _ in columns:
        dataset[name] = pd.to_numeric(dataset[name], errors="coerce")

    # Set column types and remove NaNs
    dataset.dropna(inplace=True)
    for name, col_type in columns:
        dataset[name] = dataset[name].astype(col_type)

    return dataset


def write_output_csv(filename: str, predictions):
    with open(filename, 'r') as file:
        file.write("\n".join(predictions))
