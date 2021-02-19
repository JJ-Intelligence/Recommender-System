import argparse

from models.matrix_fact import MatrixFactoriser
from src.io_handler import read_train_csv, read_test_csv, write_output_csv
from src.models.knn import KNNModel


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--trainfile', type=str,
                        help='File containing the train data')
    parser.add_argument('--testfile', type=str,
                        help='File containing the test data')
    parser.add_argument('--outputfile', type=str,
                        help='File to output predictions')
    args = parser.parse_args()

    print("Starting reading CSVs")
    train_dataset, evaluation_dataset = read_train_csv(args.trainfile, test_size=0.9999)
    test_dataset = read_test_csv(args.testfile)

    print("Starting training")
    model = MatrixFactoriser(k=10, hw_init=0.1)
    model.train(train_dataset, eval_dataset=evaluation_dataset, epochs=1, lr=0.001)

    print("Starting predicting")
    predictions = model.predict(test_dataset)

    print("Writing output")
    write_output_csv(args.outputfile, predictions)

    print("Final evaluation")
    evaluation = model.eval(evaluation_dataset)
    print(evaluation)


if __name__ == "__main__":
    main()
