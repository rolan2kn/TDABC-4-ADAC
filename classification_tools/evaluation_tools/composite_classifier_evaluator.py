#!/usr/bin/python
# -*- coding: utf-8 -*-

import time
import os

from classification_tools.evaluation_tools.classifier_evaluator \
    import ClassifierEvaluator
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from utilities.register import Register

"""
CompositeClassifierEvaluator class

This class is intended to be for assess the classification_tools used in this project.
As our method is a Repeated K-Fold Cross Validation we need to 
collect all metrics corresponding to each pass of this method. 
And provide all graphical and tabular interpretations of results. 
"""
class CompositeClassifierEvaluator(ClassifierEvaluator):
    def __init__(self, **kwargs):
        super(CompositeClassifierEvaluator, self).__init__(**kwargs)
        self.evaluator_list = []

    def set_method_name(self, method_name):
        self.method_name = method_name

    """
    add_evaluator method aims is to incorporate a new runs of our classifier to the list
    
    :param expected_results are the real labels of each dataset sample
    :param predicted_results are the labels assigned by our classifier 
    """
    def add_evaluator(self, clsfr_evaluator):
        if clsfr_evaluator is None:
            return

        self.evaluator_list.append(clsfr_evaluator)

    def save_metrics(self):
        pass

    def get_values(self):
        pass

    def get_classifier_list(self):
        return self.evaluator_list

    def get_general_metrics(self):
        pass

    def plot_generalized_confusion_matrix(self, cmap = plt.cm.Blues):
        # print all CM

        size = len(self.evaluator_list)
        if size > 5:
            cols = int(np.ceil(np.sqrt(size)))
            rows = int((size + cols-1)/cols)
        else:
            cols, rows = size, 1

        heigth = rows**2 if rows > 2 else rows*3.5
        fig, axs = plt.subplots(int(rows), int(cols), constrained_layout=True
                               ,figsize=(cols**2, heigth))

        fig.suptitle("Confusion Matrices", size=22)
        
        for idx in range(size):
            i = int(idx / cols)
            j = int(idx % cols)
            if len(axs.shape) == 1:
                ax = axs[idx]
            else:
                ax = axs[i,j]
            self.evaluator_list[idx].plot_generalized_confusion_matrix(
                cmap=cmap, ax=ax, save=False, size=10)

        self.save_picture("confusion matrices")
        plt.close(fig)

    def plot_roc_auc_and_ap_curve(self):
        size = len(self.evaluator_list)

        curves_info = {}
        for idx in range(size):
            info = self.evaluator_list[idx].plot_roc_auc_curve(save=False)
            avp_info = self.evaluator_list[idx].plot_precision_recall_curve(save=False)

            info.update(avp_info)
            curves_info.update({"{0}-{1}".format(self.evaluator_list[idx].classifier_name, self.evaluator_list[idx].method_name): info})

        self.plot_bars(curves_info)

        return curves_info

    def acc_curve(self, by_class=False):
        labels = ["Trials (Repeated Cross-Validation Executions)", "Accuracy"]

        Xs = range(len(self.evaluator_list[0].metrics_list))
        Ys = []

        for i, clsf_ev in enumerate(self.evaluator_list):
            _Y = [metric.general_metrics.accuracy()
                   for metric in clsf_ev.metrics_list]
            Ys.append(_Y)

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels, title="accuracy curve")

    def precision_curve(self, by_class=False):
        labels = ["Trials (Repeated Cross-Validation Executions)", "Precision"]

        Xs = range(len(self.evaluator_list[0].metrics_list))
        Ys = []

        for i, clsf_ev in enumerate(self.evaluator_list):
            _Y = [metric.general_metrics.precision()
                  for metric in clsf_ev.metrics_list]
            Ys.append(_Y)

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels, title="precision curve")

    def recall_curve(self, by_class=False):
        labels = ["Trials (Repeated Cross-Validation Executions)", "Recall"]

        Xs = range(len(self.evaluator_list[0].metrics_list))
        Ys = []

        for i, clsf_ev in enumerate(self.evaluator_list):
            _Y = [metric.general_metrics.recall()
                  for metric in clsf_ev.metrics_list]
            Ys.append(_Y)

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels, title="recall curve")

    def fp_rate_curve(self, by_class=False):
        labels = ["Trials (Repeated Cross-Validation Executions)", "FP Rate"]

        Xs = range(len(self.evaluator_list[0].metrics_list))
        Ys = []

        for i, clsf_ev in enumerate(self.evaluator_list):
            _Y = [metric.general_metrics.false_positive_rate()
                  for metric in clsf_ev.metrics_list]
            Ys.append(_Y)

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels, title="false positive rate curve")

    def tn_rate_curve(self, by_class=False):
        labels = ["Trials (Repeated Cross-Validation Executions)", "TN Rate"]

        Xs = range(len(self.evaluator_list[0].metrics_list))
        Ys = []

        for i, clsf_ev in enumerate(self.evaluator_list):
            _Y = [metric.general_metrics.true_negative_rate()
                  for metric in clsf_ev.metrics_list]
            Ys.append(_Y)

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels, title="true negative rate curve")

    def fpd_rate_curve(self, by_class=False):
        labels = ["Trials (Repeated Cross-Validation Executions)", "False Positive Discovery Rate"]

        Xs = range(len(self.evaluator_list[0].metrics_list))
        Ys = []

        for i, clsf_ev in enumerate(self.evaluator_list):
            _Y = [metric.general_metrics.false_positive_discovery_rate()
                  for metric in clsf_ev.metrics_list]
            Ys.append(_Y)

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels, title="false positive discovery rate curve")

    def f1_curve(self, by_class=False):
        labels = ["Trials (Repeated Cross-Validation Executions)", "F1-Measure"]
        Xs = range(len(self.evaluator_list[0].metrics_list))
        Ys = []

        for i, clsf_ev in enumerate(self.evaluator_list):
            _Y = [metric.general_metrics.f1_measure()
                  for metric in clsf_ev.metrics_list]
            Ys.append(_Y)

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels, title="f1 measure curve")

    def fb_curves(self, by_class=False):

        Xs = range(len(self.evaluator_list[0].metrics_list))
        Ys = {}

        if len(self.evaluator_list) == 0:
            return

        metric = self.evaluator_list[0].metrics_list[0]
        fb = metric.general_metrics.fb_measure()
        Ys = {b:[[] for _ in range(len(self.evaluator_list))] for b in fb}

        for i, clsf_ev in enumerate(self.evaluator_list):
            for metric in clsf_ev.metrics_list:
                fbi = metric.general_metrics.fb_measure()

                for b in fbi:
                    Ys[b][i].append(fbi[b])

        for b in Ys:
            self.plot_2D_curve(Xs=Xs, Ys=Ys[b], labels=["Trials (Repeated Cross-Validation Executions)",
                                                        "F{0}-Measure".format(b)], title="f{0} measure curve".format(b))

        del Ys

    def classificaton_error_curve(self, by_class=False):
        labels = ["Trials (Repeated Cross-Validation Executions)", "Classification Error"]
        Xs = range(len(self.evaluator_list[0].metrics_list))
        Ys = []

        for i, clsf_ev in enumerate(self.evaluator_list):
            _Y = [metric.general_metrics.classification_error()
                  for metric in clsf_ev.metrics_list]
            Ys.append(_Y)

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels,
                                  title="classification error curve", limits=(-1.3, 1.3))

    def balanced_acc_curve(self, by_class=False):
        labels = ["Trials (Repeated Cross-Validation Executions)", "Balanced Accuracy"]
        Xs = range(len(self.evaluator_list[0].metrics_list))
        Ys = []

        for i, clsf_ev in enumerate(self.evaluator_list):
            _Y = [metric.general_metrics.balanced_accuracy()
                  for metric in clsf_ev.metrics_list]
            Ys.append(_Y)

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels, title="balanced accuracy curve")

    def mcc_curve(self, by_class=False):
        labels = ["Trials (Repeated Cross-Validation Executions)", "MCC"]
        Xs = range(len(self.evaluator_list[0].metrics_list))
        Ys = []

        for i, clsf_ev in enumerate(self.evaluator_list):
            _Y = [metric.general_metrics.mattews_corr_coef()
                  for metric in clsf_ev.metrics_list]
            Ys.append(_Y)

        return self.plot_2D_curve(Xs, Ys, labels, "mathews correlation coeficient curve", limits=(-1.5, 1.5))

    def gmean_curve(self, by_class=False):
        labels = ["Trials (Repeated Cross-Validation Executions)", "Geometric Mean"]
        Xs = range(len(self.evaluator_list[0].metrics_list))
        Ys = []

        for i, clsf_ev in enumerate(self.evaluator_list):
            _Y = [metric.general_metrics.geometric_mean()
                  for metric in clsf_ev.metrics_list]
            Ys.append(_Y)

        return self.plot_2D_curve(Xs, Ys, labels, "geometric mean curve")

    def plot_bars(self, curves_info):
        labels = list(curves_info.keys())
        no_clfs = len(labels)

        x = np.arange(no_clfs)  # the label locations
        width = 0.4  # the width of the bars

        fig, ax = plt.subplots(1, 1, figsize = (16, 9))

        value_list = []
        bar_labels = []
        for l in curves_info[labels[0]]:
            value_list.append([0]*no_clfs)
            bar_labels.append(l)

        for idx, ckey in enumerate(curves_info):
            clsf = curves_info[ckey]

            for jdx, l in enumerate(clsf):
                value_list[jdx][idx] = round(clsf[l]["micro"], 2)

        # Add some text for labels, title and custom x-axis tick labels, etc.

        # ax.legend(loc="center right", bbox_to_anchor=(-0.2, 0., 0, 0))
        for i in range(no_clfs):
            parts = labels[i].split("-")
            if parts[0].find("TDABC") != -1:
                labels[i] = "{0}-{1} ".format(parts[0], parts[-1][0])
            else:
                labels[i] = "{0}".format(parts[0])


        for i, vlist in enumerate(value_list):
            xp = x-width/2-0.02 if i%2 == 0 else x+width/2+0.02
            rect = ax.bar(xp, vlist, width, label=bar_labels[i])
            ax.bar_label(rect, padding=8)

        ax.set_ylabel('Scores')
        ax.set_title('AUC and AP measures', fontsize=25)
        ax.set_xticks(x)
        ax.set_xticklabels(labels=labels, fontsize=15)
        ax.legend(loc=4, bbox_to_anchor=(-0.2, 0., 0, 0), fontsize=15)
        fig.tight_layout()
        self.save_picture("AUC AP measures")
        plt.close("all")

    def plot_2D_curve(self, Xs, Ys, labels, title, rows = 1, cols = 1,
                      fig = None, ax = None, flabels=None, save=True, limits=None):

        size = 6
        if fig is None or ax is None:
            fig, ax = plt.subplots(nrows=rows, ncols=cols,constrained_layout=True)
        max_v = len(Ys)
        if flabels is None:
            flabels = [clsf.method_name for clsf in self.evaluator_list]

        for idx, Y in enumerate(Ys):
            ax.plot(Xs, Y, color=plt.cm.jet(np.float(idx) / max_v),
                    linewidth=2, label=flabels[idx])

        ax.set_title(title, size=size+8)
        inf = -0.5
        sup = 1.5

        if limits is not None and len(limits) > 1:
            inf = limits[0]
            sup = limits[1]

        ax.set(ylim=(inf, sup))

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.legend(loc="center right", bbox_to_anchor=(-0.2, 0., 0, 0))  # Add little spacing around the legend box)
        #plt.tight_layout()
        if save:
            self.save_picture(title)
        plt.close()

    def plot_all(self):
        try:
            results = self.plot_roc_auc_and_ap_curve()

            self.acc_curve()
            self.recall_curve()
            self.precision_curve()
            self.tn_rate_curve()
            self.fp_rate_curve()
            self.fpd_rate_curve()
            self.f1_curve()
            self.fb_curves()
            self.classificaton_error_curve()
            self.balanced_acc_curve()
            self.mcc_curve()
            self.gmean_curve()

            self.plot_generalized_confusion_matrix()
            self.save_metrics()
            plt.close("all")
            self.cleanup()

            return results
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def save_picture(self, title):
        file_name = self.build_paths(title, "png")
        plt.savefig(file_name)
        plt.close(plt.gcf())

    def cleanup(self):
        for clsf in self.evaluator_list:
            clsf.cleanup()

        del self.evaluator_list
