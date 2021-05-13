from collections import defaultdict

import numpy as np

from data import EvaluationDataset
from models import ModelBase
from progress_bars import EpochBar
from data import TestDataset, TrainDataset


class KNNModel(ModelBase):

    def __init__(self):
        super().__init__()
        self.mu = None
        self.users = {}
        self.items = defaultdict(set)

    def initialise(self, *args, **kwargs):
        pass

    @staticmethod
    def cosine_similarity(a, b):
        s = a_sum = b_sum = 0
        for item in {**a, **b}.keys():
            s += a.get(item, 0) * b.get(item, 0)
            a_sum += a.get(item, 0)
            b_sum += b.get(item, 0)
        return s/(a_sum*b_sum)

    def train(self, dataset: TrainDataset, *args, **kwargs):
        user_ids = dataset.X["user id"].to_numpy()
        item_ids = dataset.X["item id"].to_numpy()
        ratings = dataset.y["user rating"].to_numpy()

        # Fill user-item-ratings map
        print("Processing")
        for i in range(len(user_ids)):
            # Add rating to maps
            if user_ids[i] not in self.users:
                self.users[user_ids[i]] = {item_ids[i]: ratings[i]}
            else:
                self.users[user_ids[i]][item_ids[i]] = ratings[i]

            self.items[item_ids[i]].add(user_ids[i])

        # Calculate global average
        self.mu = np.mean(ratings)

    def train_step(self, dataset: TrainDataset, eval_dataset: EvaluationDataset, *args, **kwargs):
        return self.train(dataset)

    def predict(self, dataset: TestDataset):
        user_ids = dataset.dataset["user id"].to_numpy()
        item_ids = dataset.dataset["item id"].to_numpy()

        print("Predicting")
        with EpochBar('Processing', max=len(user_ids) // 10_000) as bar:
            results = []
            for i in range(len(user_ids)):
                bar.next()
                unweighted = 0
                total_weights = 0

                # Find users who've rated that item
                if user_ids[i] in self.users:
                    for user_b in self.items[item_ids[i]]:
                        sim = self.cosine_similarity(self.users[user_ids[i]], self.users[user_b])
                        unweighted += sim * self.users[user_b][item_ids[i]]
                        total_weights += sim

                results.append(unweighted/total_weights if total_weights > 0 else self.mu)

            return np.array(results)

    def save(self, checkpoint_dir):
        pass

    def load(self, checkpoint_dir):
        pass
