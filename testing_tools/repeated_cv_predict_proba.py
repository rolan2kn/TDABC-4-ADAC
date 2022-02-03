#!/usr/bin/python
# -*- coding: utf-8 -*-

from testing_tools.repeated_cross_validation import RepeatedCrossValidation


class RepeatedCVPredictProba(RepeatedCrossValidation):
    def __init__(self, **kwargs):
        super(RepeatedCVPredictProba, self).__init__(**kwargs)

    def make_predictions(self, classifier, X):
        return classifier.predict_proba(X)


