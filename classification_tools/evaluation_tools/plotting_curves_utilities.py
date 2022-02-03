#!/usr/bin/python
# -*- coding: utf-8 -*-
from itertools import cycle

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

from utilities.register import Register

class CurveHandler:
    def __init__(self, **kwargs):
        y = kwargs.get("y", None)
        self.classes = kwargs.get("classes", None)
        self.y_score = np.array(kwargs.get("y_score", None))
        self.fig_name = kwargs.get("fig_name", None)
        self.title = kwargs.get("title", 'Receiver operating characteristic')
        self.save = kwargs.get("save", True)

        if y is None or self.y_score is None:
            raise Exception("ERROR: we expect two collections of labels, the real and predicted"
                            " label lists")
        classes = np.unique(y)
        if len(classes) == 2:
            self.y_score = np.array([[np.argmax(yy)] for yy in self.y_score])
            self.y_test = label_binarize(y, classes=classes, pos_label=1)
        else:
            self.y_test = label_binarize(y, classes=classes)
        self.n_classes = self.y_test.shape[1]

    def execute(self):
        pass

    def draw_plots(self, **kwargs):
        pass


class RocAucCurveHandler(CurveHandler):
    """
    Code of Receiver Operating Characteristic (ROC) based on sklearn examples

    .. note::

        See also :func:`sklearn.metrics.roc_auc_score`,
                 :ref:`sphx_glr_auto_examples_model_selection_plot_roc_crossval.py`

    """
    def __init__(self, **kwargs):
        title = kwargs.get("title", 'Receiver operating characteristic')
        kwargs.update({"title": title})
        super(RocAucCurveHandler, self).__init__(**kwargs)

    def execute(self):
        try:
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(self.n_classes):
                fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i], self.y_score[:, i])
                # roc_auc[i] = auc(fpr[i], tpr[i])
                roc_auc[i] = roc_auc_score(self.y_test[:, i], self.y_score[:, i],
                                           multi_class="ovr", average="weighted")

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(self.y_test.ravel(), self.y_score.ravel())
            # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            roc_auc["micro"] = roc_auc_score(self.y_test.ravel(), self.y_score.ravel(), multi_class="ovr",
                                             average="weighted")

            # ..........................................
            # Compute macro-average ROC curve and ROC area

            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(self.n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= self.n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            if self.save:
                self.draw_plots(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

            results = {}
            for i in roc_auc:
                results.update({i: roc_auc[i]})

            return {"AUC": results}
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def draw_plots(self, **kwargs):
        fpr = kwargs.get("fpr", None)
        tpr = kwargs.get("tpr", None)
        roc_auc = kwargs.get("roc_auc", None)
        if fpr is None or tpr is None or roc_auc is None:
            return
        # %%
        lw = 2

        # Plot all ROC curves
        # fig = plt.Figure(figsize=(10, 7)
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        c_max = self.n_classes+2
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color=plt.cm.jet(np.float(0) / c_max), linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color=plt.cm.jet(np.float(1) / c_max), linestyle=':', linewidth=4)

        for i in range(self.n_classes):
            plt.plot(fpr[i], tpr[i], color=plt.cm.jet(np.float(2+i) / c_max), lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([-0.1, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('{0} Multi-Class'.format(self.title))
        # plt.legend(loc=(0, -.4), prop=dict(size=7))
        plt.legend()
        if self.fig_name is None:
            plt.show()
        else:
            plt.savefig("{0}_MC.png".format(self.fig_name))
            plt.close(plt.gcf())
        plt.close("all")


class PrecRecCurveHandler(CurveHandler):
    """
    Code of Precision Recall Curve based on sklearn examples

    See https://scikit-learn.org/stable/auto_examples/model_selection
    /plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py

    """
    def __init__(self, **kwargs):

        title = kwargs.get("title", 'Precision Recall Curve')
        kwargs.update({"title": title})
        super(PrecRecCurveHandler, self).__init__(**kwargs)

    def execute(self):
        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()

        for i in range(self.n_classes):
            precision[i], recall[i], _ = precision_recall_curve(self.y_test[:,i],
                                                                self.y_score[:, i])
            average_precision[i] = average_precision_score(self.y_test[:, i], self.y_score[:, i], average="weighted")

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(self.y_test.ravel(),
                                                                        self.y_score.ravel())
        average_precision["micro"] = average_precision_score(self.y_test, self.y_score,
                                                             average="micro")

        if self.save:
            self.draw_plots(precision=precision, recall=recall, average_precision=average_precision)

        results = {}
        for i in average_precision:
            results.update({i: average_precision[i]})

        return {"AVP": results}


    def draw_plots(self, **kwargs):
        precision = kwargs.get("precision", None)
        recall = kwargs.get("recall", None)
        average_precision = kwargs.get("average_precision", None)
        if precision is None or recall is None or average_precision is None:
            return

        plt.figure()
        plt.step(recall['micro'], precision['micro'], where='post')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(
            'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
                .format(average_precision["micro"]))

        if self.fig_name is None:
            plt.show()
        else:
            plt.savefig("{0}_PR_micro.png".format(self.fig_name))
            plt.close(plt.gcf())
        plt.close("all")
        # setup plot details
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

        # plt.figure(figsize=(7, 8))
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

        c_max = self.n_classes + 2
        lines.append(l)
        labels.append('iso-f1 curves')

        l, = plt.plot(recall["micro"], precision["micro"], color=plt.cm.jet(np.float(0) / c_max), lw=2)
        lines.append(l)
        labels.append('micro-average Precision-recall (area = {0:0.2f})'
                      ''.format(average_precision["micro"]))

        for i, color in zip(range(self.n_classes), colors):
            l, = plt.plot(recall[i], precision[i], color=plt.cm.jet(np.float(i+2) / c_max), lw=2)
            lines.append(l)
            labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                          ''.format(i, average_precision[i]))

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(lines, labels, loc=(0, -.4), prop=dict(size=14))

        if self.fig_name is None:
            plt.show()
        else:
            plt.savefig("{0}_PR_MC.png".format(self.fig_name))
            plt.close(plt.gcf())
        plt.close("all")
