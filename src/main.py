import argparse
import os
from collections import defaultdict

import pandas as pd

from evaluation import to_cross_validation_datasets
from models import MatrixFactoriser, RandomModel, IndustryBaselineModel
from io_handler import read_train_csv, read_test_csv, write_output_csv
from train import start_training


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('run_option', choices=['run', 'tune', 'load'])
    parser.add_argument('--trainfile', type=str, help='File containing the train data')
    parser.add_argument('--testfile', type=str, help='File containing the test data')
    parser.add_argument('--outputfile', type=str, help='File to output predictions')
    parser.add_argument('--checkpointfile', type=str, help='Checkpoint file to load')
    parser.add_argument('--model', type=str,
                        help='Model to use when \'run_option\' is \'run\' (MatrixFact/Random/Baseline)', default=None)
    args = parser.parse_args()

    if args.run_option == "tune":

        print("Reading training CSV")
        train_dataset, evaluation_dataset, test_dataset = read_train_csv(args.trainfile, test_size=0.1, eval_size=0.1)

        print("\n---- Starting training ----")
        an = start_training(train_dataset, evaluation_dataset)
        print("\n---- Finished training ----")

        print("\nBest trial:")
        print(an.best_trial)
        print("\nBest checkpoint:")
        print(an.best_checkpoint)
        print("\nBest config:")
        print(an.best_config)
        print("\nBest result:")
        print(an.best_result)

        print("\n---- Loading best checkpoint model ----")
        model = MatrixFactoriser()
        model.load(os.path.join(an.best_checkpoint, "checkpoint.npz"))

        print("MSE on test dataset")
        print(model.eval(test_dataset))

        print("Reading prediction dataset")
        predict_dataset = read_test_csv(args.testfile)

        print("Creating predictions")
        predictions = model.predict(predict_dataset)

        print("Writing prediction output")
        write_output_csv(args.outputfile, predict_dataset, predictions)

    elif args.run_option == "run":
        if args.model is None:
            # Default to matrix fact
            model_name = "matrixfact"
        else:
            model_name = args.model.lower()

        print("Reading training CSV")
        train_dataset, test_dataset = read_train_csv(args.trainfile, test_size=0.1, eval_size=0)

        print("Starting training")
        if model_name == 'matrixfact':
            model = MatrixFactoriser()
            model.initialise(k=32, hw_init_stddev=0.014676120289293371)
            model.train(
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                epochs=70,
                batch_size=16_384,
                lr=0.0068726720195871754,
                user_reg=0.0676216799448991,
                item_reg=0.06639363622316222,
                user_bias_reg=0.12389941928866091,
                item_bias_reg=0.046243201501061273,
            )

        # model.save("model.npz")

        elif model_name == 'average':
            model = RandomModel(is_normal=False)

            print("Starting training")
            model.train(
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
            )

        elif model_name == 'random':
            model = RandomModel(is_normal=True)

            print("Starting training")
            model.train(
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
            )

        elif model_name == 'baseline':
            print("Training baseline model")
            model = IndustryBaselineModel()
            model.initialise()
            model.train(train_dataset)
        else:
            raise RuntimeError("Invalid argument for 'run_model'")

        print("Run on test set")
        evaluation = model.eval(test_dataset)
        print(evaluation)

        print("Reading prediction dataset")
        predict_dataset = read_test_csv(args.testfile)

        print("Creating predictions")
        predictions = model.predict(predict_dataset)

        print("Writing prediction output")
        write_output_csv(args.outputfile, predict_dataset, predictions)

    elif args.run_option == "load":
        print("Loading model from:", args.checkpointfile)
        model = MatrixFactoriser()
        model.load(args.checkpointfile)

        print("Reading prediction dataset")
        predict_dataset = read_test_csv(args.testfile)

        print("Reading training CSV to test MSE")
        train_dataset, test_dataset = read_train_csv(args.trainfile, test_size=1)
        ev = model.eval(test_dataset)
        print("MSE on training set:", ev.mse)

        print("Creating predictions")
        predictions = model.predict(predict_dataset)

        print("Writing prediction output")
        write_output_csv(args.outputfile, predict_dataset, predictions)

    elif args.run_option == "evaluate":
        models = [
            (
                # Our model
                MatrixFactoriser,
                {"k": 32, "hw_init_stddev": 0.014676120289293371},
                {"epochs": 70, "batch_size": 16_384, "lr": 0.0068726720195871754, "user_reg": 0.0676216799448991,
                 "item_reg": 0.06639363622316222, "user_bias_reg": 0.12389941928866091,
                 "item_bias_reg": 0.046243201501061273},
            ), (
                # Random model with a normal distribution
                RandomModel, {"is_normal": True}, {},
            ), (

            )
        ]

        print("Loading dataset")
        dataset = read_train_csv(args.trainfile, test_size=0., eval_size=0.)

        cv_results = defaultdict(list)
        for cv_num, (train_dataset, test_dataset) in enumerate(
                to_cross_validation_datasets(dataset, n_splits=5, seed=1)):

            for model_cls, init_kwargs, train_kwargs in models:
                # Run model on CV fold
                print("Evaluating '%s' on CV fold %d" % (model_cls.__name__, cv_num))
                model = model_cls()
                model.initialise(**init_kwargs)
                model.train(train_dataset, **train_kwargs)
                evaluation = model.eval(test_dataset)
                print("> Results:\n", evaluation)

                # Update results
                cv_results["CV fold"].append(cv_num)
                cv_results["Model"].append(model_cls.__name__)
                for k, v in evaluation.__dict__.items():
                    cv_results[k].append(v)

        # Output a CSV
        cv_df = pd.DataFrame(cv_results)
        print("Final evaluation results:\n", cv_df.to_markdown())
        cv_df.to_csv(args.outputfile)


if __name__ == "__main__":
    main()
