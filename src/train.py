import os
from datetime import datetime
from math import inf

from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB

from models.matrix_fact import MatrixFactoriser


TRAINING_ITERATIONS = 200
CHECKPOINT_FREQ = 10  # How frequently to save checkpoints


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
                hw_init_stddev=config["hw_init_stddev"]
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

        # if prev_mse < ev.mse and prev_prev_mse < ev.mse:
        #     break

        prev_prev_mse, prev_mse = prev_mse, ev.mse


def start_training(train_dataset, evaluation_dataset):
    """
    :param train_dataset:
    :param evaluation_dataset:
    :return: tune.ExperimentAnalysis
    """

    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=TRAINING_ITERATIONS,
        reduction_factor=3,
        mode="min",
        metric="mse")

    bohb_search = TuneBOHB(
        # space=config_space,  # If you want to set the space manually
        max_concurrent=2,
        mode="min",
        metric="mse",
        points_to_evaluate=[
            {
                # Starting point to evaluate
                "k": 4,
                "hw_init_stddev": 0.5,
                "user_reg": 0.1,
                "item_reg": 0.1,
                "batch_size": 131072,
                "lr": 0.01
            }
        ])

    return tune.run(
        tune.with_parameters(custom_trainable, data=(train_dataset, evaluation_dataset)),
        name="recommender-system"+str(datetime.now()).replace(":", "-").replace(".", "-"),
        local_dir="results/",
        config={
            "model_type": "matrix_fact",
            "k": tune.choice([1, 2, 4, 8, 16, 32]),
            "hw_init_stddev": tune.uniform(0, 1),
            "user_reg": tune.uniform(0, 0.5),
            "item_reg": tune.uniform(0, 0.5),
            "batch_size": tune.choice([16384, 32768, 65536, 131072]),
            "lr": tune.loguniform(0.001, 0.05)
        },
        # resources_per_trial={
        #  "cpu": 2
        # },
        verbose=3,
        scheduler=bohb_hyperband,
        search_alg=bohb_search,
        keep_checkpoints_num=5,
        num_samples=6,
        time_budget_s=int(3600*47.5)
    )
