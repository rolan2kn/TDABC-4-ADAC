#!/usr/bin/python
# -*- coding: utf-8 -*-

import random
import sys

import numpy as np
from sklearn.base import BaseEstimator

from dataset_tools.dataset_handler import DatasetHandler
from simplicial_complex_tools.persistence_interval_stage import PersistenceIntervalStage
from utilities.register import Register

from simplicial_complex_tools.simplicial_complex_builder \
    import FilteredSimplicialComplexBuilder
from simplicial_complex_tools.simplicial_complex_types \
    import SimplicialComplexType
from simplicial_complex_tools.persistence_interval_selector \
    import NaivePersistenceIntervalSelector
from classification_tools.tda_based_classifiers.selector_types \
    import SelectorTypeHandler


class TDABasedClassifier(BaseEstimator):
    def __init__(self, **kwargs):
        self.algorithm_mode = kwargs.get("algorithm_mode", FilteredSimplicialComplexBuilder.DIRECT)
        self.dataset_handler = kwargs.get("dataset_handler", None)
        if self.dataset_handler is None:
            self.dataset_handler = kwargs.get("data_handler", None)

        self.complex_type = kwargs.get("complex_type", SimplicialComplexType())
        self.selector_handler = kwargs.get("selectors", None)
        if self.selector_handler is None:
            self.selector_handler = kwargs.get("selector_type", SelectorTypeHandler())
        self.pi_stage_handler = PersistenceIntervalStage(kwargs.get("pi_stage", PersistenceIntervalStage.DEATH))

        self.kwargs = None
        self.moment = 0
        self.simplex_tree = None

    '''
    get_link calcula el link(sigma) debido a que gudhi no computa esta funcion
    '''
    def ignore_simplex_on_link(self, simplex, _fval):
        return len(simplex) == 0

    def get_link(self, sigma):
        """
        as gudhi SimplexTree dont have link method
        :param sigma:
        :return:
        """

        if self.simplex_tree is None or \
                sigma in (None, set(), [], ()):
            return [], []

        link2 = []
        fv = []
        if not (type(sigma) in (set, list, tuple)):
            sigma = [sigma]
        try:
            _star = self.simplex_tree.get_star(sigma)

            for idx, data in enumerate(_star):                # _ is the filtration value, its not necessary here
                simplex, _fval = data
                simplex = set(simplex).difference(sigma)

                if self.ignore_simplex_on_link(simplex, _fval):
                    continue
                fv.append(_fval)

                # link = link.union(list(simplex)) # use a list and do append instead of using a set
                link2.append(list(simplex))

            del _star
        except BaseException as e:
            Register.add_error_message(e)
            print("ERROR en get_link: {0}".format(e))

        return link2, fv

    '''
     Let sigma be an unknown point which it not belongs to simplicial complex K. 
     Let epsilon be the filtration value defined by the selected persistence interval. 
     The extended Star(sigma) is the union of the Star(tau) of all 0-simplices in K which 
     are inside of an open ball B(sigma, 2*epsilon) centered in sigma with radius 2*epsilon.
     
     Let T = {\tau \in K | dim(tau) = 0; ||sigma-tau|| <= 2*epsilon } the set of all 0-simplices in K inside 
     the open ball B(sigma, tau)
     
     eSt(sigma) = {St(\tau)}_{\tau \in T}
     
     We use the extended Star because we need to include the 0-simplices closest to sigma. In case all 
     closest 0-simplices are unlabeled points then the extended star will 
     get the union of Lk(sigma) because no label contribution will be collect from points in T. 
    '''
    def get_extended_star(self, sigma, epsilon):
        if self.simplex_tree is None or \
                sigma in (None, set(), [], ()):
            return [], []

        sigma_point = np.array(self.dataset_handler.get_test_point(sigma))
        S1, _ = self.dataset_handler.get_training_info()
        S = np.array(S1)
        del S1
        del _
        Sdiff = np.linalg.norm(S - sigma_point, axis=1)
        elems = (Sdiff <= 2*epsilon)  # self.moment is the selected epsilon value
        values, = np.where(elems)

        del S
        #del Sdiff

        eSt = []
        fv = []

        for i in values:
            fv.append(Sdiff[i])
            eSt.append([i])
            continue

            # TODO: to assess the difference between take the label directly and ask for the star or neighbors
            _star = self.simplex_tree.get_star([i])
            for idx, data in enumerate(_star):  # _ is the filtration value, its not necessary here
                simplex, _fval = data

                if self.ignore_simplex_on_link(simplex, _fval):
                    continue

                if _fval == 0:
                    _fval = Sdiff[simplex[0]]

                fv.append(_fval)
                eSt.append(simplex)

            del _star
        del elems
        del Sdiff

        return eSt, fv

    '''
    Psi es la funcion de asignacion que hace corresponder un conjunto de etiquetas t \in P(T) a cada simplice sigma \in K
    '''
    def label_propagation(self, sigma):
        up_votes = self.upward_label_propagation(sigma)
        down_votes = self.downward_label_propagation(sigma)

        result = up_votes + down_votes

        return result

    def empty_contribution_vector(self):
        return np.zeros(len(self.dataset_handler.tags_set))

    def compute_local_weight(self, filtration_value):
        return 1 / (filtration_value or 1)

    def compute_contributions(self, simplex_collection, fvalue_collection):
        result = self.empty_contribution_vector()

        if simplex_collection is None \
                or len(simplex_collection) == 0:
            return result

        for id, alpha in enumerate(simplex_collection):
            weight = self.compute_local_weight(fvalue_collection[id])
            result += self.downward_label_propagation(alpha) * weight

        return result

    def post_processing_upwards(self, **kwargs):
        return kwargs.get("partial_results", self.empty_contribution_vector())

    def upward_label_propagation(self, sigma):
        try:
            result = self.empty_contribution_vector()

            if sigma is None:
                return result

            link, fv = self.get_link(sigma)

            if len(link) == 0:                 # case 1: Lk(v) == \emptyset
                link, fv = self.get_extended_star(sigma, self.moment)

            result = self.compute_contributions(link, fv)

            if sum(result) == 0 and len(link) > 0:  # case 2
                result = self.post_processing_upwards(sigma=sigma, fvaues=fv, link=link)

            del link
            del fv

            return result
        except Exception as e:
            Register.add_error_message(e)
            raise "Este error fue en TDABC::upward_label_propagation {0}".format(e)

    def downward_label_propagation(self, sigma):
        try:
            result = self.empty_contribution_vector()

            if sigma is None:
                return result

            sigma_tag = self.dataset_handler.get_tag_from_training(sigma)

            if sigma_tag is not None:  # sigma belongs to the training set
                tid = self.dataset_handler.get_pos_from_tag(sigma_tag)
                result[tid] += 1

                return result

            if isinstance(sigma, int) or \
                    not (type(sigma) in (list, tuple, np.array)):
                sigma = [sigma]

            if len(sigma) == 1:  # then ksimplex \in X and t = None
                return result
            else:
                for tau in sigma:
                    result += self.downward_label_propagation(tau)

            return result
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def Card(self, sigma):
        try:
            return len(sigma) if type(sigma) == list or type(sigma) == tuple else 1
        except Exception as e:
            Register.add_error_message(e)
            raise e

    '''
    La funcion Gamma retorna un vector V, donde cada elemento 
    v_i \in V representa la cantidad de apariciones (o votos) obtenidos por la etiqueta 
    t_i \in T durante el calculo de Psi(\sigma).  
    '''
    def Gamma(self, sigma):
        try:
            V = self.label_propagation(sigma)
            #print("Lista de Clasificacion: {0}".format(V))

            return V
        except Exception as e:
            Register.add_error_message(e)
            raise e

    # Upsilon asigna a sigma la etiqueta con mayor cantidad de votos
    def Upsilon(self, sigma, prob=False):
        try:
            V = self.Gamma(sigma)
            if prob:
                total = np.sum(V)
                if total == 0:
                    total = 1
                nV = V / total

                return nV
            else:
                i = self.M(V)
                l = self.dataset_handler.get_label_from_pos(i)

                return l
            return None
        except Exception as e:
            Register.add_error_message(e)
            raise e

    """
    M es una función que dado un vector V ∈ R^L 
    devuelve un entero 0 <= i < L, donde i es la posicion de la componente de V 
    con valor máximo
    Si hay mas de un valor maximo se escoge aleatoriamente.
    """
    def M(self, vector):
        try:
            size = len(vector)
            if size < 1:
                Register.add_info_message("No label founded!! wrong vector size. We will give erroneously the label-0")
                print("No label founded!! wrong vector size. We will give erroneously the label-0")
                return 0

            major = max(vector)
            if major == 0:
                Register.add_info_message("No label founded!! we will select one uniformly at random!!!")
                print("No label founded!! we will select one uniformly at random!!!")

                return random.choice(range(len(self.dataset_handler.tags_set)))

            pos = []

            for idx, element in enumerate(vector):
                if major == element:
                    pos.append(idx)

            del major

            return random.choice(pos)
        except Exception as e:
            Register.add_error_message(e)
            raise e

    """
    Build a filtered simplicial complex on P = S \cup X
    perform the persistence homology and returns all diag
    """
    def build_filtered_simplicial_complex(self, data=None):
        try:
            if self.kwargs is None:
                self.kwargs = {}
            filtered_sc_builder = FilteredSimplicialComplexBuilder(
                data_handler=self.dataset_handler, complex_type=self.complex_type,
                data=data, builder_type=self.algorithm_mode, **self.kwargs)

            self.simplex_tree = filtered_sc_builder.execute()
            diag = self.simplex_tree.persistence()

            self.kwargs.update({"diag": diag, "simplex_tree": self.simplex_tree})
            Register.add_info_message("Persistence by default")
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def get_desired_persistence_interval(self, choice):
        try:
            naive_pinterval_selector = NaivePersistenceIntervalSelector(self.simplex_tree)

            pi = None
            if self.selector_handler.is_Average(choice):
                pi = naive_pinterval_selector.closest_to_average_lifetime()
            elif self.selector_handler.is_HAverage(choice):
                pi = naive_pinterval_selector.closest_to_harmonic_mean_lifetime()
            elif self.selector_handler.is_GAverage(choice):
                pi = naive_pinterval_selector.closest_to_geometric_mean_lifetime()

            elif self.selector_handler.is_Maximal(choice):
                pi = naive_pinterval_selector.max_lifetime_interval()
            elif self.selector_handler.is_Randomized(choice):
                pi = naive_pinterval_selector.upper_half_randomized_lifetime()

            dim = naive_pinterval_selector.get_corresponding_dimension(pi)

            return pi, dim
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def destroy(self):
        del self.simplex_tree
        self.simplex_tree = None

    def select_life_stage(self, selector):
        try:
            persistence_interval, dim = \
                self.get_desired_persistence_interval(choice=selector)

            if persistence_interval is None or len(
                    persistence_interval) == 0:  # we ignore the process
                self.destroy()
                raise Exception("Bad Interval selection")

            pi = persistence_interval[0]
            pe = persistence_interval[1]
            stage = self.pi_stage_handler.get_stage()

            if self.pi_stage_handler.is_Birth(stage):
                moment = pi
            elif self.pi_stage_handler.is_Death(stage):
                moment = pe
            elif self.pi_stage_handler.is_Middle(stage):
                moment = (pi + pe) * 0.5

            self.moment = moment        # we save the current epsilon

            self.kwargs.update({'selector': self.selector_handler.to_str(selector),
                                'interval': persistence_interval,
                                'pi': moment,
                                'dim': dim})

            return moment, persistence_interval, dim
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def set_params(self, **params):
        if self.kwargs is None:
            self.kwargs = params
        else:
            self.kwargs.update(params)

    def fit(self, *args, **kwargs):
        if len(args) > 0 and self.dataset_handler is None:
            self.dataset_handler = DatasetHandler()
            self.dataset_handler.load_dataset(args[0], args[1])
        self.set_params(**kwargs)

    def predict_proba(self, X=None, data=None):
        return self.predict(X, prob=True, data=data)

    def predict(self, X=None, prob=False, data=None):
        try:
            if not X or self.kwargs is None:
                return None

            can_i_draw = self.kwargs.get("can_i_draw", False)
            selector = self.kwargs.get("selector", SelectorTypeHandler.RANDOMIZED)

            self.build_filtered_simplicial_complex(data=data)  # to compute simplicial complex and filtrations

            pe, pinterval, dim = self.select_life_stage(selector)

            Register.add_info_message("Selected Persistence interval {1}: {0}".
                                      format(self.pi_stage_handler.to_str(), pe))


            self.simplex_tree.prune_above_filtration(pe)
            Register.add_info_message("Prune above persistence interval {0}".format(self.pi_stage_handler.to_str()))

            predicted_values = []

            for idx in X:
                try:
                    value = self.Upsilon(idx, prob)
                    predicted_values.append(value)
                except Exception as e:
                    Register.add_error_message("{0}::=> point {1}".format(e, idx))
                    raise e

            self.destroy()

            return predicted_values
        except Exception as e:
            Register.add_error_message(e)
            raise e
