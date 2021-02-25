import os
from datetime import datetime
from math import inf
import numpy as np

from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.stopper import TrialPlateauStopper

from models.matrix_fact import MatrixFactoriser


TRAINING_ITERATIONS = 100
CHECKPOINT_FREQ = 5  # How frequently to save checkpoints
EARLY_TERMINATION = False


def custom_trainable(config, data, checkpoint_dir=None):
    def save_checkpoint(x):
        with tune.checkpoint_dir(step=x) as _checkpoint_dir:
            path = os.path.join(_checkpoint_dir, "checkpoint")
            model.save(path)

    train_dataset, eval_dataset = data

    model = {
        "matrix_fact": MatrixFactoriser()
     }[config["model_type"]]

    # If loading the model from checkpoint
    if checkpoint_dir:
        model.load(os.path.join(checkpoint_dir, "checkpoint"))
    else:
        if config["model_type"] == "matrix_fact":
            model.initialise(
                k=config["k"],
                hw_init=config["hw_init"]
            )
    prev_mse = prev_prev_mse = inf

    for i in range(1, TRAINING_ITERATIONS+1):
        ev = model.train_step(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            lr=config["lr"],
            batch_size=config["batch_size"],
            user_reg=config["user_reg"],
            item_reg=config["item_reg"],
        )

        if i % CHECKPOINT_FREQ == 0:
            save_checkpoint(i)

        tune.report(mse=ev.mse)

        # If the MSE starts increasing, stop training and save checkpoint
        if prev_prev_mse < prev_mse < ev.mse and EARLY_TERMINATION:
            save_checkpoint(i)
            break

        prev_prev_mse, prev_mse = prev_mse, ev.mse


def start_training(train_dataset, evaluation_dataset, time_budget=3600):
    """
    :param time_budget:
    :param train_dataset:
    :param evaluation_dataset:
    :return: tune.ExperimentAnalysis
    """

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=5,
        hyperparam_mutations={
            # distribution for resampling
            "lr": lambda: np.random.uniform(0.0001, 1),
            # allow perturbations within this set of categorical values
            "user_reg": tune.uniform(1, 0.0001),
            "item_reg": tune.uniform(1, 0.0001),
            "k": [2, 4, 6, 8]
        })

    return tune.run(
        tune.with_parameters(custom_trainable, data=(train_dataset, evaluation_dataset)),
        name="recommender-system"+str(datetime.now()).replace(":", "-").replace(".", "-"),
        scheduler=scheduler,
        local_dir="results/",
        metric="mse",
        mode="min",
        # stop=TrialPlateauStopper(metric="mse", std=0.002),
        config={
            "model_type": "matrix_fact",
            "k": tune.choice([2, 4, 6, 8]),
            "hw_init": 0.1,
            "batch_size": 100_000,
            "lr": tune.uniform(0.001, 0.1),
            "user_reg": tune.uniform(0.001, 0.1),
            "item_reg": tune.uniform(0.001, 0.1),
        },
        resources_per_trial={
         "cpu": 1
        },
        verbose=3,
        keep_checkpoints_num=4,
        checkpoint_score_attr="mse",
        num_samples=4,
        time_budget_s=time_budget
    )
