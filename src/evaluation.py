from src import ModelABC


class Evaluation:
    def __init__(self, model: ModelABC):
        self.model = model

    def get_mse(self) -> float:
        pass

    def __str__(self):
        pass
