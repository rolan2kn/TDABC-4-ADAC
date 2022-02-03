#!/usr/bin/python
# -*- coding: utf-8 -*-
import time
import os


import numpy as np
import sklearn
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from classification_tools.evaluation_tools.plotting_curves_utilities import RocAucCurveHandler, PrecRecCurveHandler
from classification_tools.evaluation_tools.result_storer import ResultHDF5Storer
from utilities import utils
from utilities.register import Register
from classification_tools.evaluation_tools.evaluation_metrics_handler import EvaluationMetricsHandler

"""
ClassifierEvaluator class

This class is intended to be for assess the classification_tools used in this project.
As our method is a Repeated K-Fold Cross Validation we need to 
collect all metrics corresponding to each pass of this method. 
And provide all graphical and tabular interpretations of results. 
"""
class ClassifierEvaluator:
    def __init__(self, **kwargs):
        self.classifier_name = kwargs.get("classifier_name", "")
        #self.classifier_name = self.classifier_name.replace("_", "\_")
        self.method_name = kwargs.get("method_name", "")
        #self.method_name = self.method_name.replace("_", "\_")
        self.metrics_list = []
        self.general_metrics = None
        self.classes = kwargs.get("classes", [])
        self.dataset_name = kwargs.get("dataset_name", "")
        self.time_preffix = kwargs.get("time_preffix", "")
        self.graphics_by_class = kwargs.get("graphics_by_class", True)
        self.folder = kwargs.get("folder", "docs/CLASSIFIER_EVALUATION/")
        self.path = kwargs.get("path", None)
        self.space = (1.05, 1)
        self.result_storer = None

    def set_method_name(self, method_name):
        self.method_name = method_name

    def get_clasifier_fullname(self):
        short_name = self.short_method_name()

        full_name = self.classifier_name.split("-")[0]

        if len(short_name) > 0:
            full_name = "{0}-{1}".format(full_name, short_name)

        return full_name

    """
    add_metrics method aims is to incorporate a new runs of our classifier to the list
    
    :param expected_results are the real labels of each dataset sample
    :param predicted_results are the labels assigned by our classifier 
    """
    def add_metrics(self, expected_results, predicted_results):
        if expected_results is None or len(expected_results) == 0:
            print("Warning no hay expected results solo predicted ggg")
            Register.add_warning_message("no hay expected results solo predicted ggg")
            expected_results = list(predicted_results)
        if predicted_results is None or len(predicted_results) == 0:
            print("ERROR no hay predicted results esto es grave")
            Register.add_error_message("no hay predicted results esto es grave")
            return

        if self.classes is None or len(self.classes) == 0:
            self.classes = np.unique(expected_results)

        metrics = EvaluationMetricsHandler(expected_results,
                                           predicted_results, self.classes, self.path)
        metrics.compute_metrics()
        self.metrics_list.append(metrics)

    """
    build_paths method aims is to construct all paths to store the assessment results
    
    :param title is the real name of the document to store
    :param extension is the extension of the document
    
    :returns the document full name to store  
    """
    def build_paths(self, title, extension=""):
        if self.path:
            path = self.path
        else:
            if len(self.dataset_name) > 0:
                path = "{0}/{2}/{1}/".format(
                    utils.get_module_path(), self.dataset_name, self.folder)
            else:
                path = "{0}/{1}/".format(
                    utils.get_module_path(), self.folder)

            if self.time_preffix is not None and len(self.time_preffix) > 0:
                path += "{0}/".format(self.time_preffix)

        if len(self.method_name) > 0:
            path += "{0}/".format(self.method_name)

        if not os.path.exists(path):
            os.makedirs(path)
        dot = ".{0}".format(extension) if len(extension) > 0 else ""
        if self.time_preffix is not None and len(self.time_preffix) > 0:
            file_name = time.strftime(
                "{0}{1}_{2}{3}".format(path, self.classifier_name,
                                        title, dot))
        else:
            file_name = time.strftime(
            "{0}%y.%m.%d__%H.%M.%S_{1}_{2}{3}".format(path,
                                                       self.classifier_name,
                                                       title, dot))

        return file_name

    def save_metrics(self, **kwargs):
        file_name = self.build_paths("metrics", "txt")
        auc = kwargs.get("auc_dict", {})
        avg = kwargs.get("avgp_dict", {})

        general_metrics = self.get_general_metrics()
        general_metrics.set_auc_avg_prec(auc=auc, avg_p=avg)
        general_metrics.save_to_file(file_name)

    def get_values(self):
        if self.result_storer is None:
            y_true = []
            y_pred = []

            for metrics in self.metrics_list:
                size = len(metrics.expected_results)
                for i in range(size):
                    y_true.append(metrics.expected_results[i])
                    y_pred.append(metrics.predicted_results[i])

            self.result_storer = ResultHDF5Storer(expected_results=y_true,
                                                  predicted_results=y_pred,
                                                  storage_path=self.path)
            del y_pred
            del y_true

        return self.result_storer.get_expected_results(), self.result_storer.get_predicted_results()

    def get_general_metrics(self):
        if self.general_metrics is None:
            real_y, pred_y = self.get_values()
            self.general_metrics = EvaluationMetricsHandler(real_y, pred_y, self.classes, self.path)
            self.general_metrics.compute_metrics()

        return self.general_metrics

    def short_method_name(self):
        if self.method_name.find("RANDOMIZED") != -1 or \
                self.method_name.find("MAXIMAL") != -1 or \
                self.method_name.find("AVERAGE") != -1:
            return self.method_name[0]

        if self.method_name.find("OUTSIDE") != -1:
            return self.method_name
        return self.method_name

    def plot_generalized_confusion_matrix(self, cmap = plt.cm.Blues, ax=None, save=True, size=16):
        """
        This function prints and plots the confusion matrix.
        the generalized confusion matrix compute the full predicted results over full real results
        """
        #matplotlib.get_cachedir()
        y_true, y_pred = self.get_values()
        if type(y_pred[0]) == np.ndarray:
            y_pred = [np.argmax(y) for y in y_pred]

        title = '{0}-{1}'.format(
            self.classifier_name,
            self.short_method_name())

        # Only use the labels that appear in the data
        np_clss = np.array(self.classes)
        classes = []
        uclss = unique_labels(y_true, y_pred)
        for idx in uclss:
            classes.append(idx)           # classes[unique_labels(y_true, y_pred)]

        # Compute confusion matrix
        if sklearn.__version__.find("23") != -1:
            cm = confusion_matrix(y_true, y_pred, normalize='true', labels=classes)
        else:
            cm = confusion_matrix(y_true, y_pred, labels=classes)
            cm = np.array(cm)
            d = np.sum(cm[:], axis=1)
            cm = np.array([cm[i]/d[i] for i in range(len(cm))])

        print(cm)
        if ax is None:
            fig, ax = plt.subplots(1, 1,  constrained_layout=True,
                               figsize=(16,9))

        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        # ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes)
        plt.rc('xtick', labelsize=size)
        plt.rc('ytick', labelsize=size)
        ax.set_xlabel('Predicted label', fontsize=size)
        ax.set_ylabel('True label', fontsize=size)
        ax.set_title(title, fontsize=size)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        # thresh = cm.max() / 2.
        thresh = 0.5
        sizet = str(size + size // 8)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center", size=sizet,
                        color="white" if cm[i, j] > thresh else "black")
        # plt.tight_layout()
        if save:
            self.save_picture(title)

    def plot_roc_auc_curve(self, save=True):
        y_true, y_pred = self.get_values()
        if type(y_pred[0]) == np.ndarray:
            return RocAucCurveHandler(y=y_true, y_score=y_pred, fig_name=self.build_paths("ROC_AUC"), save=save).execute()

        return {}

    def plot_precision_recall_curve(self, save=True):
        y_true, y_pred = self.get_values()
        if type(y_pred[0]) == np.ndarray:
            return PrecRecCurveHandler(y=y_true, y_score=y_pred, fig_name=self.build_paths("Precision_recall"), save=save).execute()

        return {}

    def acc_curve(self):
        Xs = range(len(self.metrics_list))
        Ys = [[metric.general_metrics.accuracy() for metric in self.metrics_list]]
        labels = ["Trials", "Accuracy"]

        if self.graphics_by_class:
            for label in self.classes:
                Ys.append([])
                for metric in self.metrics_list:
                    label_metrics = metric.general_metrics[label]
                    Ys[-1].append(label_metrics.accuracy())

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels,
                                  title="accuracy curve")

    def precision_curve(self):
        Xs = range(len(self.metrics_list))
        Ys = [[metric.general_metrics.precision()
               for metric in self.metrics_list]]
        labels = ["Trials", "Precision"]

        if self.graphics_by_class:
            for label in self.classes:
                Ys.append([])
                for metric in self.metrics_list:
                    label_metrics = metric.general_metrics[label]
                    Ys[-1].append(label_metrics.precision())

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels,
                                  title="precision curve")

    def recall_curve(self):
        Xs = range(len(self.metrics_list))
        Ys = [[metric.general_metrics.recall() for metric in self.metrics_list]]
        labels = ["Trials", "Recall"]

        if self.graphics_by_class:
            for label in self.classes:
                Ys.append([])
                for metric in self.metrics_list:
                    label_metrics = metric.general_metrics[label]
                    Ys[-1].append(label_metrics.recall())

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels,
                                  title="recall curve")

    def fp_rate_curve(self):
        Xs = range(len(self.metrics_list))
        Ys = [[metric.general_metrics.false_positive_rate() for metric in self.metrics_list]]
        labels = ["Trials", "False Positive Rate"]

        if self.graphics_by_class:
            for label in self.classes:
                Ys.append([])
                for metric in self.metrics_list:
                    label_metrics = metric.general_metrics[label]
                    Ys[-1].append(label_metrics.false_positive_rate())

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels,
                                  title="fp rate curve")

    def tn_rate_curve(self):
        Xs = range(len(self.metrics_list))
        Ys = [[metric.general_metrics.true_negative_rate() for metric in self.metrics_list]]
        labels = ["Trials", "True Negative Rate"]

        if self.graphics_by_class:
            for label in self.classes:
                Ys.append([])
                for metric in self.metrics_list:
                    label_metrics = metric.general_metrics[label]
                    Ys[-1].append(label_metrics.true_negative_rate())

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels,
                                  title="tn rate curve")

    def fpd_rate_curve(self):
        Xs = range(len(self.metrics_list))
        Ys = [[metric.general_metrics.false_positive_discovery_rate()
               for metric in self.metrics_list]]
        labels = ["Trials", "False Positive Discovery Rate"]

        if self.graphics_by_class:
            for label in self.classes:
                Ys.append([])
                for metric in self.metrics_list:
                    label_metrics = metric.general_metrics[label]
                    Ys[-1].append(label_metrics.false_positive_discovery_rate())

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels,
                                  title="fpd rate curve")

    def f1_curve(self):
        Xs = range(len(self.metrics_list))
        Ys = [[metric.general_metrics.f1_measure() for metric in self.metrics_list]]
        labels = ["Trials", "F1-measure"]

        if self.graphics_by_class:
            for label in self.classes:
                Ys.append([])
                for metric in self.metrics_list:
                    label_metrics = metric.general_metrics[label]
                    Ys[-1].append(label_metrics.f1_measure())

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels,
                                  title="f1 measure curve")

    def fb_curve(self):
        try:
            Xs = range(len(self.metrics_list))
            Ydict = {}

            for metric in self.metrics_list:
                fb = metric.general_metrics.fb_measure()

                for b in fb:
                    if b not in Ydict:
                        Ydict.update({b: [[]]})
                        if self.graphics_by_class:
                            Ydict[b].extend([[] for _ in self.classes])
                    Ydict[b][0].append(fb[b])

            if self.graphics_by_class:
                for i, label in enumerate(self.classes):
                    for metric in self.metrics_list:
                        label_metrics = metric.general_metrics[label]
                        fb = label_metrics.fb_measure()

                        for b in fb:
                            Ydict[b][1+i].append(fb[b])

            for b in Ydict:
                self.plot_2D_curve(Xs=Xs, Ys=Ydict[b], labels=["Trials", "F{0}-measure".format(b)],
                                          title="f{0} measure curve".format(b))

            return
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def classificaton_error_curve(self):
        Xs = range(len(self.metrics_list))
        Ys = [[metric.general_metrics.classification_error() for metric in self.metrics_list]]
        labels = ["Trials", "Classification Error"]

        if self.graphics_by_class:
            for label in self.classes:
                Ys.append([])
                for metric in self.metrics_list:
                    label_metrics = metric.general_metrics[label]
                    Ys[-1].append(label_metrics.classification_error())

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels,
                                  title="clsf error curve")

    def balanced_acc_curve(self):
        Xs = range(len(self.metrics_list))
        Ys = [[metric.general_metrics.balanced_accuracy() for metric in self.metrics_list]]
        labels = ["Trials", "Balanced Accuracy Curve"]

        if self.graphics_by_class:
            for label in self.classes:
                Ys.append([])
                for metric in self.metrics_list:
                    label_metrics = metric.general_metrics[label]
                    Ys[-1].append(label_metrics.balanced_accuracy())

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels,
                                  title="bacc curve")


    def mcc_curve(self):
        Xs = range(len(self.metrics_list))
        Ys = [[metric.general_metrics.mattews_corr_coef() for metric in self.metrics_list]]
        labels = ["Trials", "Mathews Correlation Coeficient Curve"]

        if self.graphics_by_class:
            for label in self.classes:
                Ys.append([])
                for metric in self.metrics_list:
                    label_metrics = metric.general_metrics[label]
                    Ys[-1].append(label_metrics.mattews_corr_coef())

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels, limits=(-1.3, 1.3),
                                  title="mcc curve")

    def gmean_curve(self):
        Xs = range(len(self.metrics_list))
        Ys = [[metric.general_metrics.geometric_mean() for metric in self.metrics_list]]
        labels = ["Trials", "Geometric Mean"]

        if self.graphics_by_class:
            for label in self.classes:
                Ys.append([])
                for metric in self.metrics_list:
                    label_metrics = metric.general_metrics[label]
                    Ys[-1].append(label_metrics.geometric_mean())

        return self.plot_2D_curve(Xs=Xs, Ys=Ys, labels=labels,
                                  title="gmean curve")

    def plot_2D_curve(self, Xs, Ys, labels, title, color=None,
                      marker=None, rows = 1, cols = 1, limits=None,
                      fig = None, ax = None, save=True):
        try:
            if fig is None or ax is None:
                fig, ax = plt.subplots(nrows=rows, ncols=cols)
            max_v = len(Ys)
            if len(Ys) == 1:
                glabels = [title]

            else:
                glabels = ["Overall"]
                glabels.extend(["label {0}".format(i) for i in self.classes])

            for idx, Y in enumerate(Ys):
                xsize = len(Xs)
                ysize = len(Y)
                Y.extend([0]*(xsize-ysize))
                if color is not None and marker is not None:
                    ax.plot(Xs, Y, color=plt.cm.jet(np.float(idx) / max_v),
                            linewidth=2, label=glabels[idx])
                else:
                    ax.plot(Xs, Y, linewidth=2, label=glabels[idx])

            fig.suptitle(title)
            inf = -0.5
            sup = 1.5
            if limits is not None and len(limits) > 1:
                inf = limits[0]
                sup = limits[1]

            ax.set(ylim=(inf, sup))
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.legend(loc="center right", borderaxespad=0.2,
                      bbox_to_anchor=self.space,
                      )  # Add little spacing around the legend box)
            plt.subplots_adjust(right=0.85)
            if save:
                self.save_picture(title)
            plt.close("all")
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def plot_all(self):
        auc_dict = self.plot_roc_auc_curve()
        avgp_dict = self.plot_precision_recall_curve()

        self.acc_curve()
        self.recall_curve()
        self.precision_curve()
        self.tn_rate_curve()
        self.fp_rate_curve()
        self.fpd_rate_curve()
        self.f1_curve()
        self.fb_curve()
        self.classificaton_error_curve()
        self.balanced_acc_curve()
        self.mcc_curve()
        self.gmean_curve()

        self.plot_generalized_confusion_matrix()
        self.save_metrics(auc_dict=auc_dict, avgp_dict=avgp_dict)
        plt.close("all")

    def save_picture(self, title):

        file_name = self.build_paths(title, "png")
        plt.savefig(file_name)
        plt.close(plt.gcf())

    def cleanup(self):
        if self.result_storer is not None:
            self.result_storer.remove()
            del self.result_storer

        if self.general_metrics is not None:
            self.general_metrics.cleanup()
            self.general_metrics = None

        for metric in self.metrics_list:
            metric.cleanup()

        del self.metrics_list
