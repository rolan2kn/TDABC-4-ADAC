#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import SVC


class SVMClassifier:
    def __init__(self, **kwargs):
        self.training = kwargs.get("training", None)
        self.training_tags = kwargs.get("training_tags", None)
        self.linear = kwargs.get("linear", False)
        self.dataset_handler = kwargs.get("dataset_handler", None)
        if self.dataset_handler is not None:
            ttraining, ttags = self.dataset_handler.get_training_info()
            self.training = ttraining
            self.training_tags = ttags
        self.unique_tags = np.unique(self.training_tags)

    def predict(self, new_data):
        class_weight = {c: self.dataset_handler.imbalanced_ratio[c] for c in self.unique_tags}

        # import some data to play with
        if self.linear:
            clf = SVC(kernel="linear", C=0.025, class_weight=class_weight)
        else:
            clf = SVC(gamma=2, C=1, class_weight=class_weight)

        clf.fit(self.training, self.training_tags)  # entrena el dataset con las
                                                    # etiquetas...

        result = []
        for value in new_data:
            cx1 = clf.predict(np.array([value]))

            result.append(self.get_label(cx1[0]))

        return result

    def predict_proba(self, new_data):
        class_weight = {c: self.dataset_handler.imbalanced_ratio[c] for c in self.unique_tags}

        # import some data to play with
        if self.linear:
            clf = SVC(kernel="linear", C=0.025, class_weight=class_weight, probability=True)
        else:
            clf = SVC(gamma=2, C=1, class_weight=class_weight, probability=True)

        clf.fit(self.training, self.training_tags)  # entrena el dataset con las
                                                    # etiquetas...

        result = clf.predict_proba(np.array(new_data))

        return result

    def get_label(self, yid):
        if self.dataset_handler is None:
            return yid

        return self.dataset_handler.get_label_from_pos(yid)
