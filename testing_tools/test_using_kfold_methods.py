import numpy as np
from numpy import mean, std
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, KFold, GroupKFold, cross_val_score
from sklearn.datasets import make_moons
# test classification dataset
from sklearn.datasets import make_classification

from classification_tools.tda_based_classifiers.selector_types import SelectorTypeHandler
from classification_tools.tda_based_classifiers.tdabc_using_link_propagation import \
    TDABasedClassifierUsingLinkPropagation
from simplicial_complex_tools.persistence_interval_stage import PersistenceIntervalStage
from simplicial_complex_tools.simplicial_complex_builder import FilteredSimplicialComplexBuilder
from simplicial_complex_tools.simplicial_complex_types import SimplicialComplexType


class TestUsingKFoldMethods:
    def __init__(self):
        pass

    def get_model(self):
        params = {"complex_type":SimplicialComplexType(type=SimplicialComplexType.RIPS, max_dim=3, max_value=1),
            "algorithm_mode":FilteredSimplicialComplexBuilder.DIRECT, "pi_stage":PersistenceIntervalStage.DEATH}
        model = TDABasedClassifierUsingLinkPropagation(**params)

        return model

    def execute(self):
        X, y = make_moons(noise=0.352, random_state=1, n_samples=100)
        self.test_kfold(X, y)

    def test_kfold(self, X, y):
        cv = KFold(n_splits=10, random_state=1, shuffle=True)
        # create model
        model = self.get_model()
        # evaluate model
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        # report performance
        print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

    def test_repeated_stratified_kfold(self, X, y):
        rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=36851234)

        # create model
        model = self.get_model()
        # evaluate model
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=rskf, n_jobs=-1)

        # report performance
        print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

    def test_repeated_kfold(self, X, y):
        rskf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=36851234)

        # create model
        model = self.get_model()
        # evaluate model
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=rskf, n_jobs=-1)

        # report performance
        print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))