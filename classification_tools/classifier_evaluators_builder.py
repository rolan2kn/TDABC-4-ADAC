#!/usr/bin/python
# -*- coding: utf-8 -*-
from classification_tools.evaluation_tools.composite_classifier_evaluator import *
from classification_tools.classifier_types \
    import ClassifierTypeHandler
from classification_tools.tda_based_classifiers.selector_types import SelectorTypeHandler


class ClassifierEvaluatorsBuilder:
    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "RIPS_3d")
        self.dataset_handler = kwargs.get("dataset_handler", None)
        self.tdabc_selectors = kwargs.get("selectors", SelectorTypeHandler())
        self.evaluators = {}
        clsf_types = kwargs.get("classifier_types",
                                               ClassifierTypeHandler())
        if type(clsf_types) == int:
            self.ctype_hdlr = ClassifierTypeHandler(type=clsf_types)
        else:
            self.ctype_hdlr = clsf_types

    def execute(self):
        time_preffix = self.dataset_handler.get_time()
        dataset_name = self.dataset_handler.dataset_name()
        classes = self.dataset_handler.tags_set
        selector_names = self.tdabc_selectors.all_to_str()

        overall = CompositeClassifierEvaluator(classifier_name="OVERALL-{0}".format(self.name),
                                           method_name="GLOBAL_VIEW",
                                           time_preffix=time_preffix,
                                           dataset_name=dataset_name,
                                           classes=classes,
                                               path=self.dataset_handler.get_path())

        for ctype in self.ctype_hdlr.classifier_types:
            if ctype in self.evaluators:
                del self.evaluators[ctype]

            ctype_str = self.ctype_hdlr.to_str(ctype)
            method_names = [ctype_str]

            self.evaluators.update({ctype: None})
            if self.ctype_hdlr.is_tdabc_based(ctype):
                method_names = selector_names
                self.evaluators[ctype] = {}

            for i, mname in enumerate(method_names):
                clsfEv = ClassifierEvaluator(
                    classifier_name="{0}-{1}".format(ctype_str, self.name),
                                           method_name=mname,
                                           time_preffix=time_preffix,
                                           dataset_name=dataset_name,
                                           classes=classes, path=self.dataset_handler.get_path())

                if self.ctype_hdlr.is_baseline(ctype):
                    self.evaluators[ctype] = clsfEv
                else:
                    self.evaluators[ctype].update(
                        {self.tdabc_selectors.get_selector(i): clsfEv})
                overall.add_evaluator(clsfEv)

        return self.evaluators, overall

    def create_classifier(self, ctype, selector_name):
        time_preffix = self.dataset_handler.get_time()
        dataset_name = self.dataset_handler.dataset_name()
        classes = self.dataset_handler.tags_set

        ctype_str = self.ctype_hdlr.to_str(ctype)

        clsfEv = ClassifierEvaluator(
            classifier_name="{0}-{1}".format(ctype_str, self.name),
                                   method_name=selector_name,
                                   time_preffix=time_preffix,
                                   dataset_name=dataset_name,
                                   classes=classes, path=self.dataset_handler.get_path())

        return clsfEv