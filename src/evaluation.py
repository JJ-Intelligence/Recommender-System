from dataclasses import dataclass


@dataclass
class Evaluation:
    mse: float
    accuracy: float
    f1: float

    def __str__(self):
        return "\n".join(f"{k}:{v}" for k, v in self.__dict__.items())
