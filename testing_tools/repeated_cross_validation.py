#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import random

from simplicial_complex_tools.persistence_interval_stage import PersistenceIntervalStage
from utilities.register import Register

from simplicial_complex_tools.simplicial_complex_builder import FilteredSimplicialComplexBuilder
from classification_tools.classifier_evaluators_builder \
    import ClassifierEvaluatorsBuilder
from classification_tools.classifiers_builder import ClassifierBuilder
from classification_tools.classifier_types \
    import ClassifierTypeHandler
from simplicial_complex_tools.simplicial_complex_types \
    import SimplicialComplexType

"""
This class implements Repeated Cross Validation

"""
class RepeatedCrossValidation:
    NORMAL, EXTREME, HYPER_EXTREME = range(3)

    def __init__(self, **kwargs):
        data_handler = kwargs.get("data_handler", None)
        algorithm_mode = kwargs.get("algorithm_mode", FilteredSimplicialComplexBuilder.DIRECT)
        complex_type = kwargs.get("complex_type", None)
        fold_sequence = kwargs.get("fold_sequence", RepeatedCrossValidation.NORMAL)
        classifier_ev = kwargs.get("classifier_ev", ClassifierTypeHandler.TDABC | ClassifierTypeHandler.KNN |
                                   ClassifierTypeHandler.WKNN)
        pi_stage = kwargs.get("pi_stage", None)
        self.can_i_draw = kwargs.get("can_i_draw", False)
        self.distance = kwargs.get("metric", "euclidean")

        self.algorithm_mode = algorithm_mode
        if complex_type is None:
            complex_type = SimplicialComplexType()
        self.complex_type = complex_type
        self.dataset_handler = data_handler
        self.fold_sequence = fold_sequence
        self.ctype_hdlr = ClassifierTypeHandler(classifier_ev)
        self.selector_type_hdlr = kwargs.get("selector_type", self.ctype_hdlr.get_selector_handlers())

        self.overall_evaluator = None
        self.pi_stage = pi_stage

    def get_classifier_evaluator_builder(self):
        name = "{0}-{1}d-{2}".format(self.complex_type.to_str(),
                                     self.complex_type.get_maximal_dimension(), self.fold_sequence_name())

        return ClassifierEvaluatorsBuilder(
            name=name,
            dataset_handler=self.dataset_handler,
            selectors=self.selector_type_hdlr,
            classifier_types= self.ctype_hdlr)

    def init_execution_data(self):
        size_data = self.dataset_handler.dataset_size()
        path = self.dataset_handler.get_path()

        classifier_ev_builder = self.get_classifier_evaluator_builder()

        self.classifier_evaluators, self.overall_evaluator = classifier_ev_builder.execute()

        self.classifier_builder = ClassifierBuilder(
            dataset_handler=self.dataset_handler,
            complex_type=self.complex_type,
            selectors=self.selector_type_hdlr,
            algorithm_mode=self.algorithm_mode,
            classifier_types=self.ctype_hdlr,
            pi_stage = self.pi_stage)

        return path, size_data

    def get_overall_evaluator(self):
        return self.overall_evaluator

    def fold_sequence_name(self):
        fs_names = ["NORMAL", "EXTREME", "HYPER_EXTREME"]

        return fs_names[self.fold_sequence]

    def get_fold_sequence(self):
        if self.fold_sequence == RepeatedCrossValidation.EXTREME:
            return [5, [60, 45, 50, 55]]
        elif self.fold_sequence == RepeatedCrossValidation.HYPER_EXTREME:
            return [5, [90, 75, 80, 85, 90]]

        return [5, [10, 20, 30]]

    def make_predictions(self, classifier, X):
        return classifier.predict(X)

    def execute_tdabc(self, fold_size, fold_position):
        try:
            selection_functions = self.selector_type_hdlr.selectors()
            tdabc_type_collection = self.ctype_hdlr.get_tdabc()

            for selector in selection_functions:

                selector_name = self.selector_type_hdlr.to_str(selector)

                log_message = "EXECUTE DATANAME={6} STAGE={7} K-FOLD SEQUENCE_TYPE={5} COMPLEX={3} DIM={4} " \
                              "SELECTOR={2} k={0}, n={1}".format(fold_size, fold_position, selector_name,
                                                                 self.complex_type.get_name(),
                                                                 self.complex_type.get_maximal_dimension(),
                                                                 self.fold_sequence_name(),
                                                                 self.dataset_handler.dataset_name(),
                                                                 PersistenceIntervalStage().to_str(self.pi_stage))

                print(log_message)
                Register.add_info_message(log_message)

                for tdabc_type in tdabc_type_collection:
                    tdabc = self.classifier_builder.build_tda_based(tdabc_type)
                    if not tdabc.selector_handler.is_selector_present(selector):
                        del tdabc
                        continue

                    X, Y = self.dataset_handler.get_tdabc_data()
                    tdabc.fit(selector=selector, KJ=[fold_size, fold_position],
                                 fold_sequence=self.fold_sequence_name(),
                                 can_i_draw=self.can_i_draw, filtration=self.selector_type_hdlr.is_Average(selector),
                              metric=self.distance)

                    predicted_values = self.make_predictions(tdabc, X=X)
                    self.classifier_evaluators[tdabc_type][selector].add_metrics(Y,
                                                    predicted_values)
                    metric = \
                        self.classifier_evaluators[tdabc_type][selector].metrics_list[-1]

                    name = self.ctype_hdlr.to_str(tdabc_type)
                    sname = self.selector_type_hdlr.to_str(selector)
                    print("\n{0}::{1} Metrics\n{2}".format(name, sname, metric))
                    del X
                    del Y
                    del predicted_values
                    del tdabc
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def execute(self, ext_values=None):
        try:
            path, size_data = self.init_execution_data()
            Register.create("registro.log", path)

            fold_seq = self.get_fold_sequence()
            n = fold_seq[0]
            m = fold_seq[1][0]
            for _ in range(n):
                for k in [m]:
                    Register.add_info_message("EXECUTING REPEATED CROSS VALIDATION")

                    fold_size = int((size_data * k + 99) // 100)  # fold_size
                                                            # it represents
                                                            # the k% of all data

                    folds = int((size_data + fold_size - 1) / fold_size)
                    for j in range(folds):
                        if ext_values is None:
                            self.dataset_handler.split_dataset(fold_size, j)
                        else:
                            self.dataset_handler.split_dataset()  # to provide classification boundaries
                            self.dataset_handler.configure_external_testing_set(ext_values)

                        self.execute_tdabc(fold_size, j)
                        self.execute_baseline()

            if ext_values is None:
                return self.evaluation_stage()
            return None
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def evaluation_stage(self):
        tdabc_type_collections = self.ctype_hdlr.get_tdabc()
        for tdabc_type in tdabc_type_collections:
            tda_classifiers = self.classifier_evaluators[tdabc_type]
            for selector in tda_classifiers:
                tda_classifiers[selector].plot_all()

        for ctype in self.classifier_evaluators:
            if self.ctype_hdlr.is_baseline(ctype):
                self.classifier_evaluators[ctype].plot_all()

        result = self.overall_evaluator.plot_all()
        del self.classifier_evaluators
        del self.overall_evaluator

        return result

    def execute_baseline(self):
        try:
            ttraining, ttags, ttest, real_values = self.dataset_handler.\
                get_baseline_data()

            baseline_types = self.ctype_hdlr.get_baselines()

            for btype in baseline_types:
                if self.ctype_hdlr.is_knn_based(btype):
                    clsf = self.classifier_builder.build_knn_based(btype)
                else:
                    clsf = self.classifier_builder.build_more_classifiers(btype)

                if clsf is None:
                    continue

                ypred = self.make_predictions(classifier=clsf, X=ttest)

                if len(ypred) > 0:
                    self.classifier_evaluators[btype].add_metrics(real_values, ypred)
                    metric = self.classifier_evaluators[btype].metrics_list[-1]
                    name = self.ctype_hdlr.to_str(btype)
                    print("\n{0} Metrics\n{1}".format(name, metric))
        except Exception as e:
            Register.add_error_message(e)
            raise e
