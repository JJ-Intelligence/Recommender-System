from dataclasses import dataclass


class Evaluation(dataclass):
    mse: float

    def __str__(self):
        pass
