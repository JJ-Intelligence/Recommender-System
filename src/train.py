import os
from datetime import datetime
from math import inf

from ray import tune

from models.matrix_fact import MatrixFactoriser


TRAINING_ITERATIONS = 500
CHECKPOINT_FREQ = 5  # How frequently to save checkpoints


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
            batch_size=config["batch_size"]
        )

        if i % CHECKPOINT_FREQ == 0:
            save_checkpoint(i)

        tune.report(mse=ev.mse)

        if prev_mse < ev.mse and prev_prev_mse < ev.mse:
            break

        prev_prev_mse, prev_mse = prev_mse, ev.mse


def start_training(train_dataset, evaluation_dataset):
    """
    :param train_dataset:
    :param evaluation_dataset:
    :return: tune.ExperimentAnalysis
    """
    return tune.run(
        tune.with_parameters(custom_trainable, data=(train_dataset, evaluation_dataset)),
        name="recommender-system"+str(datetime.now()).replace(":", "-").replace(".", "-"),
        local_dir="results/",
        metric="mse",
        mode="min",
        config={
            "model_type": "matrix_fact",
            "k": tune.grid_search([1, 2, 4, 8, 16, 32, 64, 128]),
            "hw_init": 0.1,
            "batch_size": tune.grid_search([100_000, 1000_000]),
            "lr": tune.grid_search([0.01, 0.001])
        },
        resources_per_trial={
         "cpu": 2
        },
        verbose=3
    )
