#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn import neighbors
import numpy as np


class kNNClassifier:
    def __init__(self, **kwargs):
        self.training = kwargs.get("training", None)
        self.training_tags = kwargs.get("training_tags", None)
        self.weight = kwargs.get("weight", False)
        self.dataset_handler = kwargs.get("dataset_handler", None)
        if self.dataset_handler is not None:
            ttraining, ttags = self.dataset_handler.get_training_info()
            self.training = ttraining
            self.training_tags = ttags

    def predict(self, new_data):
        n_neighbors = 15 if len(self.training) > 15 else len(self.training)

        # import some data to play with
        if self.weight:
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights="distance")
        else:
            clf = neighbors.KNeighborsClassifier(n_neighbors)
        clf.fit(self.training, self.training_tags)  # entrena el dataset con las
                                                    # etiquetas...

        result = []
        for value in new_data:
            cx1 = clf.predict(np.array([value]))

            result.append(self.get_label(cx1[0]))

        return result

    def predict_proba(self, new_data):
        n_neighbors = 15 if len(self.training) > 15 else len(self.training)

        # import some data to play with
        if self.weight:
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights="distance")
        else:
            clf = neighbors.KNeighborsClassifier(n_neighbors)
        clf.fit(self.training, self.training_tags)  # entrena el dataset con las
                                                    # etiquetas...

        result = clf.predict_proba(np.array(new_data))

        return result

    def get_label(self, yid):
        if self.dataset_handler is None:
            return yid

        return self.dataset_handler.get_label_from_pos(yid)
