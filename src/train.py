from ray import tune

from models.matrix_fact import MatrixFactoriser


TRAINING_ITERATIONS = 100


def custom_trainable(config, data):
    train_dataset, eval_dataset = data
    model = {
        "matrix_fact": MatrixFactoriser(
            dataset=train_dataset,
            k=config["k"],
            hw_init=config["hw_init"]
        )
    }[config["model_type"]]

    for x in range(TRAINING_ITERATIONS):
        ev = model.train_step(
            eval_dataset=eval_dataset,
            lr=config["lr"],
            batch_size=config["batch_size"]
        )
        tune.report(mse=ev.mse)


def start_training(train_dataset, evaluation_dataset):
    tune.run(
        tune.with_parameters(custom_trainable, data=(train_dataset, evaluation_dataset)),
        name="recommender-system",
        local_dir="../results",
        metric="mse",
        mode="min",
        config={
            "model_type": "matrix_fact",
            "k": tune.grid_search([16, 32, 64]),
            "hw_init": 0.1,
            "batch_size": 100_000,
            "lr": tune.grid_search([0.01, 0.001])
        },
        resources_per_trial={
         "cpu": 2
        },
        verbose=2
    )
