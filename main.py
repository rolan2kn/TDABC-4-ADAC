

from dataset_tools.dataset_transformer import DatasetTransformer
from dataset_tools.dataset_types import DatasetTypeHandler

from testing_tools.test_using_kfold_methods import TestUsingKFoldMethods
from utilities.register import Register
from testing_tools.test_krnn_rocauc import TestKRNN_ROCAUC
import argparse


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-o", "--option", help="Select option to execute 1: Multi KRNN experiment, 2: KRNN, 3: Conventional experiment ",
                            type=int)
        parser.add_argument("-d", "--dataset", help="Select the datasets",
                            type=int)
        parser.add_argument("-path", "--path", help="Define folder to save results",
                            type=str)
        parser.add_argument("-i", "--iteration", help="Iteration number for option=1",
                            type=int)
        parser.add_argument("-n", "--number_datasets", help="Number of datasets for option=1",
                            type=int)
        parser.add_argument("-m", "--metric", help="Metric to build the simplicial comples. A collection of possible "
                                                   "metrics are: ['cityblock', 'cosine', "
                                                   "'euclidean', 'l1', 'l2', 'manhattan', 'nan_euclidean','braycurtis',"
                                                   " 'canberra', 'chebyshev','correlation', 'dice', 'hamming', 'jaccard',"
                                                   " 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', "
                                                   "'russellrao', 'seuclidean', 'sokalmichener', "
                                                   "'sokalsneath', 'sqeuclidean', 'yule']",
                            type=str)
        parser.add_argument("-t", "--transformation", help="Data Transformation type to apply"
                                                   "types are: 0=NONE, SIMPLE=1, SQUARED=2, NLOG=3, NORM=4",
                            type=str)

        args = parser.parse_args()

        if args is not None:
            option = args.option if args.option is not None else 1
            dataset = args.dataset if args.dataset is not None else DatasetTypeHandler.KRNN
            path = args.path if args.path is not None else "./docs/ADAC_TDABC_RESULTS"

            iteration = args.iteration if args.iteration is not None else 3
            ndatasets = args.number_datasets if args.number_datasets else 16
            metric = args.metric if args.metric is not None else "euclidean"

        kwargs = {"option": option,
                  "dtype": dataset,
                  "folder": path,
                  "iterations": iteration,
                  "number_of_datasets": ndatasets,
                  "csv_output": True,
                  "metric": metric}

        TestKRNN_ROCAUC().execute(**kwargs)
        # TestUsingKFoldMethods().execute()

    except BaseException as e:
        Register.add_error_message(e)
        print("ERROR global: {0}".format(e))
