from abc import abstractmethod, ABC
import numpy as np
from sklearn import metrics

from data import TrainDataset, EvaluationDataset, TestDataset
from evaluation import Evaluation


class ModelBase(ABC):
    def __init__(self):
        self.initialised = False

    def is_initialised(self):
        return self.initialised

    @abstractmethod
    def initialise(self, *args, **kwargs):
        """
        Initialise the model - including taking in the dataset
        """

    def _initialise(self, *args, **kwargs):
        if self.is_initialised():
            raise Exception("Model already initialised")
        self.initialise(*args, **kwargs)
        self.initialised = True

    @abstractmethod
    def train_step(self, dataset: TrainDataset, eval_dataset: EvaluationDataset, *args, **kwargs):
        """
        Perform a single training step.
        Override this in the custom models
        """
        pass

    def _train_step(self, dataset: TrainDataset, eval_dataset: EvaluationDataset = None, *args, **kwargs):
        if not self.is_initialised():
            raise Exception("Model not yet initialised")
        self.train_step(dataset, eval_dataset, *args, **kwargs)

    @abstractmethod
    def train(self, dataset: TrainDataset, eval_dataset: EvaluationDataset = None, epochs: int = None,
              lr: float = None):
        """
        Do a number of epochs of the train_step.
        This is an alternate way of running the training
        """
        pass

    @abstractmethod
    def predict(self, dataset: TestDataset) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, checkpoint_dir):
        pass

    def _save(self, checkpoint_dir):
        if not self.is_initialised():
            self.save(checkpoint_dir)
        else:
            raise Exception("Model not yet initialised")

    @abstractmethod
    def load(self, checkpoint_dir):
        pass

    def _load(self, checkpoint_dir):
        self.load(checkpoint_dir)
        self.initialised = True

    def eval(self, evaluation_dataset: EvaluationDataset) -> Evaluation:
        test_dataset, y_true = evaluation_dataset.to_test_dataset()
        y_preds = self.predict(test_dataset)
        y_preds_rounded = np.round(2 * y_preds) / 2  # Predictions rounded to nearest 0.5

        return Evaluation(
            mae=metrics.mean_absolute_error(y_true, y_preds),
            mse=metrics.mean_squared_error(y_true, y_preds),
            rmse=metrics.mean_squared_error(y_true, y_preds, squared=False),
            accuracy=metrics.balanced_accuracy_score(y_true, y_preds_rounded),
            f1=metrics.f1_score(y_true, y_preds_rounded, average="weighted"),
        )
