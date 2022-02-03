import time

import numpy as np
import gudhi
from math import ceil

from sklearn.metrics import pairwise_distances

from simplicial_complex_tools.simplicial_complex_types \
    import SimplicialComplexType
from utilities.register import Register


class SimplicialComplexesGenerator:
    def __init__(self, data_handler):
        self.data_handler = data_handler
        self.complex = None
        self.simplex_tree = None

    def generate_rips(self, **kwargs):
        try:
            sc_type = kwargs.get("sc_type")
            data = kwargs.get("data", None)
            partitioning = kwargs.get("partitioning", False)
            distance = kwargs.get("metric", "euclidean")

            print ("METRIC: ", distance)

            S = self.data_handler.unify_dataset(data, partitioning)
            dim = self.data_handler.get_real_dimension()
            distance_matrix = pairwise_distances(S, n_jobs=-1, metric=distance)

            self.complex = gudhi.RipsComplex(distance_matrix=distance_matrix)

            max_dim = sc_type.get_maximal_dimension()
            q = max_dim if max_dim > 0 else 1
            factor = max(q, dim) / min(q, dim)

            self.simplex_tree = self.complex.create_simplex_tree(max_dimension=float(1))
            del S
            del distance_matrix
            self.simplex_tree.collapse_edges(ceil(factor))
            self.simplex_tree.expansion(max_dim)

            return self.simplex_tree
        except Exception as e:
            Register.add_error_message(e)
            raise  e

    def generate_alpha(self, ** kwargs):
        try:
            sc_type = kwargs.get("sc_type")
            data = kwargs.get("data", None)
            partitioning = kwargs.get("partitioning", False)
            distance = kwargs.get("metric", "euclidean")

            square_alpha = sc_type.get_maximum_value()
            S = self.data_handler.unify_dataset(data, partitioning)
            self.complex = gudhi.AlphaComplex(points=S)
            del S

            if square_alpha is None:
                self.simplex_tree = self.complex.create_simplex_tree()
            else:
                self.simplex_tree = self.complex.create_simplex_tree(
                                             max_alpha_square=float(square_alpha))

            return self.simplex_tree
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def generate_complex(self,**kwargs):
        try:
            sc_type = kwargs.get("sc_type", None)

            if sc_type is None:
                sc_type = SimplicialComplexType()
                kwargs.update({"sc_type":sc_type})

            if sc_type.is_RIPS():
                self.generate_rips(**kwargs)

            else: #then is sc_type.is_ALPHA():
                self.generate_alpha(**kwargs)

            return self.simplex_tree
        except Exception as e:
            Register.add_error_message(e)
            raise e