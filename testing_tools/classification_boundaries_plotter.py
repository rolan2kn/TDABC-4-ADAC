#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import random
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from sklearn import neighbors, datasets
from matplotlib.colors import ListedColormap

from classification_tools.classifier_types import ClassifierTypeHandler
from dataset_tools.dataset_handler import DatasetHandler
from dataset_tools.dataset_types import DatasetTypeHandler
from simplicial_complex_tools.persistence_interval_stage import PersistenceIntervalStage
from simplicial_complex_tools.simplicial_complex_builder import FilteredSimplicialComplexBuilder
from simplicial_complex_tools.simplicial_complex_types import SimplicialComplexType
from testing_tools.build_dataset_4_test import BuildDataset4Test
from utilities import utils
from testing_tools.repeated_cross_validation import RepeatedCrossValidation
from utilities.register import Register


class ClassificationBoundariesPlotter:
    def dirty_work(self, dataset_type):
        if hasattr(self, "dataset_handler"):
            del self.dataset_handler

        self.dataset_handler = BuildDataset4Test(dataset_type).execute()

        for __dim in [3]:
        # for __dim in range(3, 10):
                Register.add_info_message("Creating Repeated Cross Validaton")

                rips_tdabc = RepeatedCrossValidation(
                        data_handler=self.dataset_handler,
                        complex_type=SimplicialComplexType(type=SimplicialComplexType.RIPS, max_dim=__dim, max_value=1),
                        # fold_sequence=fs,
                    # classifier_ev=WTDABC3|KNN|WKNN,
                    classifier_ev=ClassifierTypeHandler.TDABC|ClassifierTypeHandler.KNN|ClassifierTypeHandler.WKNN,
                        algorithm_mode=FilteredSimplicialComplexBuilder.DIRECT, pi_stage=PersistenceIntervalStage.MIDDLE)

                #
                # cosine_tdabc = RepeatedCrossValidation(
                #     data_handler=self.dataset_handler,
                #     complex_type=SimplicialComplexType(type=COSINE, max_dim=__dim, max_value=0.8),
                #     classifier_ev=TDABC | KNN | WKNN,
                #     algorithm_mode=DIRECT, pi_stage=DEATH)

                self.draw_boundaries(rips_tdabc)

    def execute(self):
        try:
            # for i, data in enumerate([IRIS, SWISSROLL, MOON, CIRCLES, BREAST_CANCER, WINE, SPHERE, NORMAL_DIST]):
            # for i, data in enumerate([CIRCLES]):
            for i, data in enumerate([DatasetTypeHandler.SWISSROLL]):
                self.dirty_work(DatasetTypeHandler(dtype=data))
        except BaseException as e:
            print("ERROR global: {0}".format(e))
        #

    def draw_boundaries(self, repeated_cross_validation):
        X = np.array(self.dataset_handler.unify_dataset())  # we only take the first two features. We could
        self.draw_test_points = True

        # avoid this ugly slicing by using a two-dim dataset
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        v = np.c_[xx.ravel(), yy.ravel()]
        values = [[a, b] for a, b in v]

        print("values: =====>>> ", values)

        repeated_cross_validation.execute(values)
        overall_evaluator = repeated_cross_validation.get_overall_evaluator()
        ev_methods = overall_evaluator.get_classifier_list()

        training, tlabels = self.dataset_handler.get_training_info()

        titles = [ev.get_clasifier_fullname() for ev in ev_methods]
        size_evs = len(titles)+1
        ########################

        cm = plt.cm.RdBu
        colors = [plt.cm.RdBu(np.float(l) / 2) for l in range(2)]
        cm_bright = ListedColormap(colors)
        values = np.array(values)
        plt.figure(figsize=(15, 5))
        plt.title("CLASSIFICATION BOUNDARIES")

        ax = plt.subplot(1, size_evs, 1)
        xxx = np.array(training)

        # plt.figure(figsize=(10, 5))

        Z = np.array(tlabels)

        plt.scatter(xxx[:, 0], xxx[:, 1], c=np.array(colors)[tlabels],
                    edgecolors="k", cmap=cm_bright, label=tlabels)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title("{0}".format(self.dataset_handler.dataset_name()))

        plt.tight_layout()
        size_v = len(values)
        _, y_pred = ev_methods[0].get_values()
        num_times = len(y_pred) // size_v
        select_iter = random.randint(0, num_times-1)

        for i, ev in enumerate(ev_methods):
            # Plot the predicted probabilities. For that, we will assign a color to
            # each point in the mesh [x_min, m_max]x[y_min, y_max].
            ax = plt.subplot(1, size_evs, i + 2)

            y_real, y_pred = ev.get_values()

            Z = np.array(y_pred[select_iter*size_v:size_v*(select_iter+1)])

            # Put the result into a color plot
            _Z = Z.reshape((xx.shape))
            ax.contourf(xx, yy, _Z, cmap=cm, alpha=.8)

            # Plot the training points

            ax.scatter(xxx[:, 0], xxx[:, 1], c=np.array(colors)[tlabels],
                        edgecolors="k", cmap=cm_bright, label=tlabels)

            # Plot the testing points

            if self.draw_test_points:
                ax.scatter(values[:, 0], values[:, 1], c=np.array(colors)[Z],
                            edgecolors="k", alpha=0.4, label=tlabels)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title("{0}".format(titles[i]))


        path = self.dataset_handler.get_path()
        if not os.path.isdir(path):
            os.makedirs(path)
        file_name = time.strftime(
            "{0}_BOUNDARIES_%y.%m.%d__%H.%M.%S.png".format(path))

        # plt.title("Classification Boundaries")
        # plt.legend(fontsize=20)
        plt.savefig(file_name)
        plt.close(plt.gcf())
        # plt.show()
