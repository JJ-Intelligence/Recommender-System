import argparse

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

    train_dataset, evaluation_dataset = read_train_csv(args.trainfile)
    test_dataset = read_test_csv(args.testfile)

    model = KNNModel()
    model.train(train_dataset)
    predictions = model.predict(test_dataset)

    write_output_csv(args.outputfile, predictions)

    evaluation = model.eval(evaluation_dataset)
    print(evaluation)


if __name__ == "__main__":
    main()
