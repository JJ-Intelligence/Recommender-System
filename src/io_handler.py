import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data import TrainDataset, TestDataset, EvaluationDataset

DEFAULT_RESULTS_FOLDER = "../results/"


def read_train_csv(filename: str, test_size=0.1, eval_size=0) -> (TrainDataset, EvaluationDataset, EvaluationDataset):
    dataset = _read_csv_to_dataframe(
        filename,
        [("user id", np.int32), ("item id", np.int32), ("user rating", np.float32), ("timestamp", np.float32)]
    )

    if test_size > 0 or eval_size > 0:
        x_train, x_evaluation, y_train, y_evaluation = train_test_split(
            dataset[["user id", "item id", "timestamp"]],
            dataset[["user rating"]],
            test_size=test_size,
            shuffle=True)
    if eval_size > 0:
        x_train, x_test, y_train, y_test = train_test_split(
            x_train,
            y_train,
            test_size=eval_size/(1-test_size),  # account for smaller train set
            shuffle=False)

        return \
            TrainDataset(x_train, y_train), \
            EvaluationDataset(x_evaluation, y_evaluation), \
            EvaluationDataset(x_test, y_test)
    elif test_size > 0:
        return \
            TrainDataset(x_train, y_train), \
            EvaluationDataset(x_evaluation, y_evaluation)
    else:
        return TrainDataset(dataset[["user id", "item id", "timestamp"]], dataset[["user rating"]])


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


def write_output_csv(filename: str, test_dataset: TestDataset, predictions):
    df = pd.DataFrame({
        "user id": test_dataset.dataset["user id"],
        "item id": test_dataset.dataset["item id"],
        "rating": predictions,
        "timestamp": test_dataset.dataset["timestamp"],
    })
    df.to_csv(filename, header=False, index=False)
