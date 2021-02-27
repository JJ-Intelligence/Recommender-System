import argparse
import os

from models.matrix_fact import MatrixFactoriser
from io_handler import read_train_csv, read_test_csv, write_output_csv
from train import start_training


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('run_option', choices=['run', 'tune', 'load'])
    parser.add_argument('--trainfile', type=str,
                        help='File containing the train data')
    parser.add_argument('--testfile', type=str,
                        help='File containing the test data')
    parser.add_argument('--outputfile', type=str,
                        help='File to output predictions')
    parser.add_argument('--checkpointfile', type=str,
                        help='Checkpoint file to load')
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
        write_output_csv(args.outputfile, predictions)

    elif args.run_option == "run":

        print("Reading training CSV")
        train_dataset, evaluation_dataset, test_dataset = read_train_csv(args.trainfile, test_size=0.1, eval_size=0.1)

        print("Starting training")
        model = MatrixFactoriser()
        model.initialise(k=10, hw_init_stddev=0.5)
        model.train(
            train_dataset=train_dataset,
            eval_dataset=evaluation_dataset,
            epochs=20,
            lr=0.01,
            user_reg=0.2,
            item_reg=0.2
        )

        model.save("model.npz")

        print("Run on test set")
        evaluation = model.eval(evaluation_dataset)
        print(evaluation)

        print("Reading prediction dataset")
        predict_dataset = read_test_csv(args.testfile)

        print("Creating predictions")
        predictions = model.predict(predict_dataset)

        print("Writing prediction output")
        write_output_csv(args.outputfile, predictions)

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
        write_output_csv(args.outputfile, predictions)


if __name__ == "__main__":
    main()
