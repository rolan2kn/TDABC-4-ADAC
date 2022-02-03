#!/usr/bin/python
# -*- coding: utf-8 -*-
from utilities.register import Register
from .tda_based_classifiers.tdabc_using_link_propagation import TDABasedClassifierUsingLinkPropagation
from .baseline.knn_classifier import kNNClassifier
from .baseline.svm_classifier import SVMClassifier
from .baseline.random_forest_classifier import RandomForestClassifier
from classification_tools.tda_based_classifiers.selector_types import SelectorTypeHandler

from classification_tools.classifier_types \
    import ClassifierTypeHandler


class ClassifierBuilder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tdabc_selectors = kwargs.get("selectors", None)
        if self.tdabc_selectors is None:
            self.tdabc_selectors = kwargs.get("selector_type", SelectorTypeHandler())
        self.classifiers = {}
        self.ctype_hdlr = kwargs.get("classifier_types",
                                     ClassifierTypeHandler())

    def update_argument(self, **kwargs):
        self.kwargs.update(kwargs)

    def execute(self):
        for ctype in self.classifiers:
            if self.classifiers[ctype]:
                del self.classifiers[ctype]
                self.classifiers[ctype] = None

            ctype_str = self.ctype_hdlr.to_str(ctype)
            method_names = [ctype_str]
            if self.ctype_hdlr.is_tdabc_based(ctype):
                method_names = self.tdabc_selectors.all_to_str()
                self.classifiers[ctype] = {}

            for i, mname in enumerate(method_names):
                if self.ctype_hdlr.is_tdabc_based(ctype):
                    clsf = self.build_tda_based(ctype)
                elif self.ctype_hdlr.is_knn_based(ctype):
                    clsf = self.build_knn_based(ctype)
                else:
                    clsf = self.build_more_classifiers(ctype)

                if self.ctype_hdlr.is_baseline(ctype):
                    self.classifiers[ctype] = clsf
                else:
                    self.classifiers[ctype].update(
                        {self.tdabc_selectors.get_selector(i): clsf})

        return self.classifiers

    def build_tda_based(self, ctype):
        try:
            if self.ctype_hdlr.is_ptdabc(ctype):
                tdabc_method = TDABasedClassifierUsingLinkPropagation(**self.kwargs)

            return tdabc_method
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def build_knn_based(self, ctype):
        weight = self.ctype_hdlr.is_wknn(ctype)

        self.kwargs.update({"weight": weight})

        return kNNClassifier(**self.kwargs)

    def build_more_classifiers(self, ctype):
        if self.ctype_hdlr.is_svm(ctype):
            self.kwargs.update({"linear": False})
            return SVMClassifier(**self.kwargs)
        if self.ctype_hdlr.is_lsvm(ctype):
            self.kwargs.update({"linear":True})
            return SVMClassifier(**self.kwargs)
        if self.ctype_hdlr.is_rf(ctype):
            return RandomForestClassifier(**self.kwargs)

        Register.add_error_message("Bad classifier type {1} = '{0}'".format(ctype, self.ctype_hdlr.to_str(ctype)))
        raise Exception("Bad classifier type {1} = '{0}'".format(ctype, self.ctype_hdlr.to_str(ctype)))
