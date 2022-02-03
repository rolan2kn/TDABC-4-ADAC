#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

from classification_tools.evaluation_tools.metrics \
    import Metrics, OverallMetrics

from classification_tools.evaluation_tools.result_storer import ResultHDF5Storer
from utilities.register import Register


'''
class EvaluationMetricsHandler main goal is to compute metrics for each class
from expected and predicted results, and a list of classes
'''


class EvaluationMetricsHandler:

    """
    :method __init__
    :argument expected_results are the real labels of each value x \in X,
    the training set
    :argument predicted_results are the classification resulting labels
    for each x \in X
    :argument classes is the label list
    """
    def __init__(self, expected_results, predicted_results, classes, storage_path=None):
        self.result_storer = ResultHDF5Storer(expected_results=expected_results,
                                              predicted_results=predicted_results,
                                              storage_path=storage_path)
        self.classes = classes
        self.general_metrics = OverallMetrics(self.classes)
        self.expected_results = self.result_storer.get_expected_results()
        self.predicted_results = self.result_storer.get_predicted_results()

    """
    :method init_values initialize all values to compute metrics   
    """
    def compare_labels(self, c_real, c_pred):
        if type(c_pred) == np.ndarray:
            return c_real == np.argmax(c_pred)
        return c_real == c_pred

    """
    :method: compute_metrics_per_label  
    
    predicted_list[1..N]: the list of predicted labels
    real_list[1..N]: the list of real labels
    
    We use True Positives (TP), True Negatives (TN), False Positives (FP), and 
    False Negatives (FN) per class where: 
    
    for each label l \in classes do 
        TP: predicted_list(i) iff predicted_list(i) == real_list(i) and 
                                  predicted_list(i) == l
                                  
        FP: predicted_list(i) iff predicted_list(i) != real_list(i) and 
                                  predicted_list(i) == l
        
        TN: predicted_list(i) iff predicted_list(i) == real_list(i) and 
                                  predicted_list(i) != l
                                  
        FN: predicted_list(i) iff predicted_list(i) != real_list(i) and 
                                  predicted_list(i) != l
    """

    def compute_metrics(self):

        for i, c in enumerate(self.classes):
            TP = FP = TN = FN = 0
            samples = 0
            for idx, c_pred in enumerate(self.predicted_results):  # for any predicted result
                c_real = self.expected_results[idx]

                if c_real == c:             # positive cases
                    samples += 1
                    if self.compare_labels(c_real, c_pred):
                        TP += 1
                    else:
                        FP += 1
                else:                       # negatives cases
                    if self.compare_labels(c_real, c_pred):
                        TN += 1
                    else:
                        FN += 1

            kwargs = {
                "TP": TP, "FP": FP, "TN": TN,
                "FN": FN, "SAMPLES": samples,
                "clsf_error": FP
                }

            self.general_metrics.get_metrics_from_class(c).compute_metrics(**kwargs)

        self.general_metrics.compute_metrics()

    def save_to_file(self, filename):
        self.general_metrics.to_file(filename)

    def set_auc_avg_prec(self, auc, avg_p):
        self.general_metrics.set_auc_avg_prec(auc=auc, avg_p=avg_p)

    def load_from_file(self, filename):
        self.general_metrics.from_file(filename)

    def __str__(self):
       return self.general_metrics.__str__()

    def cleanup(self):
        del self.general_metrics
        del self.expected_results
        del self.predicted_results
        self.result_storer.remove()

