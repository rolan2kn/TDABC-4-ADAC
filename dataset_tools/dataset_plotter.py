#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from matplotlib.colors import ListedColormap

from utilities.register import Register
from dataset_tools.dataset_dimensionality_reduction import DatasetDimensionalityReduction

"""
It draws a dataset
"""
class DatasetPlotter:
    def __init__(self, data_handler):
        self.data = data_handler

    def draw_data(self, **kwargs):
        try:
            if kwargs is None:
                kwargs = {}
            self.draw_general(**kwargs)
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def prepare_data(self, **kwargs):
        splitter = self.data.get_splitter()
        X = splitter.dataset
        dim = splitter.real_dimension()
        xsize = len(X)
        if xsize > 0:
            if dim > 3:
                X = DatasetDimensionalityReduction.execute(X, rtype=self.dimensionality_reduction_type)
        X = np.array(X)
        Y = np.array(splitter.tags)
        labels = np.unique(Y)

        plot_all = kwargs.get("plot_all", False)
        plot_2d = kwargs.get("plot_2d", False)

        no = len(labels)
        if plot_2d:
            no = 3 if dim > 2 else 1

        val = 5 * (no + 1)

        if not plot_all:
            no = 1
            val = 10

        max_v = max(Y)
        if max_v == 0:
            max_v = 1

        cno_sqr = 1
        fno_sqr = 1

        if plot_all:
            if plot_2d:
                labels = [labels]*no
                titles = ["XY", "XZ", "YZ"]
                seq = [[0, 1], [0, 2], [1,2]]
                cno_sqr = 1
                fno_sqr = no
            else:
                titles = ["Dataset"]
                seq = [[0, 1, 2]]
                for l in labels:
                    titles.append("label {0}".format(l))
                    seq.append([0,1,2])

                labels = [labels]
                labels.extend([int(l)] for l in labels[0])
                inc = 1 if no & 1 != 0 else 2
                cno_sqr = int(np.ceil(np.sqrt(no + inc)))
                fno_sqr = int((no + inc + cno_sqr - 1) / cno_sqr)

        kwargs.update({"X": X})
        kwargs.update({"Y": Y})
        kwargs.update({"no": no})
        kwargs.update({"val": val})
        kwargs.update({"max_v": max_v})
        kwargs.update({"dim": dim})
        kwargs.update({"cno_sqr": cno_sqr})
        kwargs.update({"fno_sqr": fno_sqr})
        kwargs.update({"labels": labels})
        kwargs.update({"titles": titles})
        kwargs.update({"seq": seq})

        return  kwargs

    def draw_general(self, **kwargs):
        """
                Draws the current dataset.
                Each label has a representative color.
                This method also, save a *.png picture of the dataset
                :return:
                """
        n = self.data.dataset_size()

        if n == 0:
            return


        self.dimensionality_reduction_type = kwargs.get("dimensionality_reduction_type",
                                                        DatasetDimensionalityReduction.UMAP)
        kwargs = self.prepare_data(**kwargs)

        X = kwargs.get("X")
        Y = kwargs.get("Y")
        no = kwargs.get("no")
        val = kwargs.get("val")
        max_v = kwargs.get("max_v")
        dim = kwargs.get("dim")
        cno_sqr = kwargs.get("cno_sqr")
        fno_sqr = kwargs.get("fno_sqr")
        labels = kwargs.get("labels")
        plot_2d = kwargs.get("plot_2d", False)
        titles = kwargs.get("titles")
        seq = kwargs.get("seq")

        w, h = plt.figaspect(0.7)
        lsize = kwargs.get("size", 10)
        fig = plt.figure(figsize=(w * no, h * no))

        plt.title(self.data.dataset_name(), size= 2*lsize*no)

        old_val = plt.rcParams['font.size']
        plt.rcParams.update({'font.size': val})

        area = kwargs.get("area", 100*no)

        title = "Dataset"
        show_label = True

        for i, ls in enumerate(labels):
            if i > 0:
                show_label = False

            if plot_2d:
                ax = plt.subplot(fno_sqr, cno_sqr, i+1)
            else:
                ax = plt.subplot(fno_sqr, cno_sqr, i + 1, projection='3d')

            ax.set_title(titles[i], size=lsize*no)
            if not plot_2d:
                ax.set_xlabel("X", size=val, labelpad=4.0+val)
                ax.set_ylabel("Y", size=val, labelpad=4.0+val)
                ax.set_zlabel("Z", size=val, labelpad=4.0+val)
            else:
                ax.set_xlabel(titles[i][0], size=val, labelpad=4.0 + val)
                ax.set_ylabel(titles[i][1], size=val, labelpad=4.0 + val)

            for l in ls:
                self.scatter_plot(ax, X, Y, l, dim, max_v, area, show_label, plot_2d, seq[i])

        fig.legend(fontsize=kwargs.get("fontsize", 8*no), loc='upper right')

        plt.savefig(self.data.get_current_name())

        plt.close(plt.gcf())
        plt.close("all")

    def scatter_plot(self, ax, X, Y, l, dim, max_v, area, show_label, plot_2d, seq):
        lb = l if show_label else None
        color = plt.cm.jet(np.float(l) / max_v)
        if not plot_2d:
            ax.scatter(X[Y == l, seq[0]], X[Y == l, seq[1]], X[Y == l, seq[2]],
                       color=color, s=area, edgecolors='k',alpha=0.8,linewidth=0.5,
                       label=lb)
        else:
            ax.scatter(X[Y == l, seq[0]], X[Y == l, seq[1]],
                       color=color, s=area, edgecolors='k',alpha=0.8,linewidth=0.5,
                       label=lb)

    def draw_hyperplanes(self, classifiers, names, scores):
        h = .02  # step size in the mesh

        #names = ["Nearest Neighbors", "TDA-Based Classifier (TDABC)"]

        figure = plt.figure(figsize=(27, 9))
        i = 1
        # iterate over datasets
        X = np.array(self.data.dataset)
        X_train = np.array(self.data.training)
        X_test = np.array(self.data.test)
        y_train = [self.data.tags_position[self.data.tags_training[i]] for i in self.data.tags_training]
        y_test = None
        if len(self.data.tags_test) > 0:
            y_test = [self.data.tags_position[self.data.tags_test[i]] for i in self.data.tags_test]

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(1, len(classifiers) + 1, i)
        ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        if y_test is not None:
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                   edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classification_tools
        for name, clf, score in zip(names, classifiers, scores):
            ax = plt.subplot(1, len(classifiers) + 1, i)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            Z = np.array(clf)

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                       edgecolors='k')
            # Plot the testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                       edgecolors='k', alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(name)
            if score is not None:
                ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                        size=15, horizontalalignment='right')

        plt.tight_layout()
        plt.show()
