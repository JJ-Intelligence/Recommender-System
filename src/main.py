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

    print("Reading training CSV")
    train_dataset, evaluation_dataset, test_dataset = read_train_csv(args.trainfile, test_size=0.1, eval_size=0.1)

    print("Starting training")
    model = MatrixFactoriser(k=10, hw_init=0.1)
    model.train(train_dataset, eval_dataset=evaluation_dataset, epochs=10, lr=0.001)

    print("Run on test set")
    evaluation = model.eval(evaluation_dataset)
    print(evaluation)

    print("Reading prediction dataset")
    predict_dataset = read_test_csv(args.testfile)

    print("Creating predictions")
    predictions = model.predict(predict_dataset)

    print("Writing prediction output")
    write_output_csv(args.outputfile, predictions)


if __name__ == "__main__":
    main()
