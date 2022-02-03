import gc
import time

import numpy as np
import matplotlib

from dataset_tools.dataset_dimensionality_reduction import DatasetDimensionalityReduction
from dataset_tools.dataset_transformer import DatasetTransformer

matplotlib.use("Agg")
from matplotlib import pyplot as plt


from classification_tools.tda_based_classifiers.selector_types import SelectorTypeHandler
from testing_tools.classification_results_formatter import ClassificationResultsFormatter
from testing_tools.repeated_cross_validation \
    import RepeatedCrossValidation
from simplicial_complex_tools.simplicial_complex_types \
    import SimplicialComplexType
from dataset_tools.dataset_types import DatasetTypeHandler
from classification_tools.classifier_types \
    import ClassifierTypeHandler

from testing_tools.repeated_cv_predict_proba import RepeatedCVPredictProba
from utilities.register import Register
from simplicial_complex_tools.simplicial_complex_builder import FilteredSimplicialComplexBuilder
import random

from simplicial_complex_tools.persistence_interval_stage import PersistenceIntervalStage
from testing_tools.build_dataset_4_test import BuildDataset4Test


class TestKRNN_ROCAUC:

    def dirty_work(self, **kwargs):

        if "folder" not in kwargs:
            # kwargs.update({"folder": "./docs/CLASSIFICATION_KRNN/EXPERIMENTS/"})
            kwargs.update({"folder": "./docs/NEW_EXPERIMENT/"})
        dataset_handler = BuildDataset4Test(**kwargs,
                                                 ).execute()

        dtype = kwargs.get("dataset_type", None)
        if dtype is not None and self.option == 1:
            current_name = dataset_handler.get_current_name()
            cpath = "{0}/".format(current_name[:-4])
            dataset_handler.set_path(cpath)
        self.path = dataset_handler.get_path()

        Register.add_info_message("#### INIT ALL EXPERIMENTS with this arguments {0}".format(str(kwargs)))

        for dim in range(3, 4):
            if dim not in self.auc_list_per_dim:
                self.auc_list_per_dim.update({dim: {}})
            for fs in [RepeatedCrossValidation.NORMAL]:
                Register.add_info_message("Creating Repeated Cross Validaton")
                if fs not in self.auc_list_per_dim[dim]:
                    self.auc_list_per_dim[dim].update({fs:{}})
                rips_tdabc = RepeatedCVPredictProba(
                    data_handler=dataset_handler,
                    # complex_type=SimplicialComplexType(type=COSINE, max_dim=dim, max_value=0.5),
                    complex_type=SimplicialComplexType(type=SimplicialComplexType.RIPS, max_dim=dim, max_value=1),
                    fold_sequence=fs,
                    selector_type=SelectorTypeHandler(SelectorTypeHandler.AVERAGE|SelectorTypeHandler.MAXIMAL|
                                                      SelectorTypeHandler.RANDOMIZED),

                    classifier_ev= ClassifierTypeHandler.PTDABC | ClassifierTypeHandler.KNN | ClassifierTypeHandler.WKNN
                                   | ClassifierTypeHandler.RF | ClassifierTypeHandler.LSVM,
                    algorithm_mode=FilteredSimplicialComplexBuilder.DIRECT, pi_stage=PersistenceIntervalStage.DEATH, **kwargs)

                results = rips_tdabc.execute()

                del rips_tdabc
                gc.collect()

                if self.option < 3:
                    classifiers = list(results.keys())
                    params = list(results[classifiers[0]].keys())
                    for param in params:
                        if param not in self.auc_list_per_dim[dim][fs]:
                            self.auc_list_per_dim[dim][fs].update({param: {}})
                            for c in classifiers:
                                if param in results[c]:
                                    self.auc_list_per_dim[dim][fs][param].update({c: [results[c][param]]})
                        else:
                            for c in classifiers:
                                self.auc_list_per_dim[dim][fs][param][c].append(results[c][param])
        del dataset_handler
        gc.collect()

    def plot_curves_info(self):

        for dim in self.auc_list_per_dim:
            for fs in self.auc_list_per_dim[dim]:
                for param in self.auc_list_per_dim[dim][fs]:
                    fig, ax = plt.subplots(1,1, figsize=(7,8))
                    plt.title(param)
                    max_v = len(self.auc_list_per_dim[dim][fs][param])
                    filled_markers = ('o', '*', 'v', '^', '<', '>', '8', 's', 'p', 'h', 'H', 'D', 'd', 'P', 'X')
                    min_v = 1
                    for idx, cls in enumerate(self.auc_list_per_dim[dim][fs][param]):
                        if type(self.auc_list_per_dim[dim][fs][param][cls][0]) != dict:
                            Y = self.auc_list_per_dim[dim][fs][param][cls]
                        else:
                            Y = [a['micro'] for a in self.auc_list_per_dim[dim][fs][param][cls] ]
                        print("{0}: {1} = {2}".format(param, cls, Y))
                        X = range(len(Y))
                        parts = cls.split("-")
                        label = parts[0] # CLASSIFIER-COMPLEX-dim-MODE_miscelanea-[AVERAGE, RANDOMIZED, MAXIMAL]
                        mode = parts[3]
                        min_v = min(np.min(Y), min_v)
                        if label.find("TDABC") != -1:
                            label = "{0}-{1}".format(label, parts[-1][0])

                        ax.plot(X, Y, color=plt.cm.jet(np.float(idx) / max_v),
                                linewidth=2, label=label, marker=filled_markers[idx])

                    ax.set_ylabel(param)
                    plt.ylim([min_v, 1])
                    # plt.xlim([0.0, 1.0])
                    ax.set_xlabel("")
                    ax.set_xticks(range(1,self.size))
                    ax.set_xticklabels(self.tick_labels, fontsize=10)
                    # ax.legend(loc = 4)
                    ax.legend()


                    name = "{0}/GLOBAL_VIEW/{1}_{2}_{3}.png".format(self.path, param, dim, mode)
                    plt.savefig(name)
                    plt.close(plt.gcf())

    def execute(self, **kwargs):
        try:
            self.auc_list_per_dim = {}
            self.option = kwargs.get("option", 1)
            time_suffix = time.strftime("%y.%m.%d_%H.%M.%S")
            dtype = kwargs.get("dtype", DatasetTypeHandler.KRNN)
            self.folder = kwargs.get("folder", "./docs/NEW_EXPERIMENT/")
            iterations = kwargs.get("iterations", 10)

            kwargs.update({"time_suffix":time_suffix, "dtype": dtype, "option": self.option,
                           "folder": self.folder, "iterations": iterations})

            if self.option == 1:
                self.multiple_krnn_experiment(**kwargs)
            elif self.option == 2:
                self.krnn_experiment(**kwargs)
            elif self.option == 3:
                self.common_experiment(**kwargs)

        except BaseException as e:
            print("ERROR global: {0}".format(e))

    def krnn_experiment(self, **kwargs):
        self.size = kwargs.get("number_of_datasets", 16) + 1
        self.tick_labels = [50 * i for i in range(1, self.size)]

        for i in range(1,self.size):
            data = kwargs.get("dtype", DatasetTypeHandler.KRNN)
            kwargs.update({"dataset_type":DatasetTypeHandler(dtype=data), "data_index":i})
            self.dirty_work(**kwargs)

        self.plot_curves_info()
        del self.auc_list_per_dim
        self.auc_list_per_dim = {}

    def multiple_krnn_experiment(self, **kwargs):
        iterations = kwargs.get("iterations", 10)
        dtype = kwargs.get("dtype", DatasetTypeHandler.KRNN)
        csv_output = kwargs.get("csv_output", iterations==0)

        for i in range(iterations):
            time_suffix = time.strftime("%y.%m.%d_%H.%M.%S")
            kwargs.update({"dataset_type": DatasetTypeHandler(dtype=dtype), "time_suffix": time_suffix})

            self.krnn_experiment(**kwargs)
            print("###### END EXPERIMENT ######")

        ClassificationResultsFormatter(root_path="{1}/".format(
            DatasetTypeHandler(dtype=dtype).to_str(), self.folder), csv_output=csv_output, desired_metrics = ["F1",
                                                              "AUC","AVP", "MCC", "GMEAN", "PREC", "REC"]).execute()
        return

    def common_experiment(self, **kwargs):
        dataset_collection = [DatasetTypeHandler.IRIS, DatasetTypeHandler.SWISSROLL, DatasetTypeHandler.MOON,
                              DatasetTypeHandler.CIRCLES, DatasetTypeHandler.BREAST_CANCER, DatasetTypeHandler.WINE,
                              DatasetTypeHandler.SPHERE, DatasetTypeHandler.NORMAL_DIST]
        self.size = len(dataset_collection)
        self.tick_labels = [DatasetTypeHandler(dtype=data).to_str() for data in dataset_collection]

        for i, data in enumerate(dataset_collection):
            kwargs.update({"dataset_type":DatasetTypeHandler(dtype=data), "data_index":i})
            self.dirty_work(**kwargs)

