import random
import math
from simplicial_complex_tools.simplicial_complexes_generator \
    import SimplicialComplexesGenerator
from utilities.register import Register


class FilteredSimplicialComplexBuilder:
    INCREMENTAL, DIRECT = range(2)
    def __init__(self, data_handler, complex_type,
                 data=None, builder_type=DIRECT, partitioning=False, **kwargs):
        self.builder = None

        if builder_type == FilteredSimplicialComplexBuilder.INCREMENTAL:
            self.builder = IncrementalSCBuilder(data_handler, complex_type, data, **kwargs)
        else:
            self.builder = DirectSCBuilder(data_handler, complex_type, data, partitioning=partitioning, **kwargs)

    def execute(self):
        return self.builder.execute()

    def predicate(self, points):
        return self.builder.predicate(points)

class IncrementalSCBuilder:
    def __init__(self, data_handler, complex_type, data, **kwargs):
        self.dataset_handler = data_handler
        self.complex_type = complex_type
        self.max_dimension = self.complex_type.get_maximal_dimension()
        self.data = data
        self.scg = None
        self.kwargs = kwargs

    def execute(self):
        if self.dataset_handler is None:
            return None
        self.scg = SimplicialComplexesGenerator(self.dataset_handler)

        Register.add_info_message(
            "Iterative generation of Simplicial Complex type: {0} and dim {1}".format(self.complex_type))
        simplex_tree = self.scg.generate_complex(sc_type=self.complex_type, partitioning=True, data = self.data, **self.kwargs)

        fvalues = self.get_filtration_values(simplex_tree)
        partitions = self.split_test_set()
        i = 0
        m = len(partitions)
        n = len(fvalues)
        V = len(self.dataset_handler.training)

        while i < m:
            X_i = partitions[i]
            j = random(0, n-1)
            e_j = fvalues[j]

            simplex_tree, V = self.add_partition(simplex_tree, X_i, V, e_j)

            i += 1

        return simplex_tree

    def get_filtration_values(self, simplex_tree):

        fvalues = []

        filtered_simplices = simplex_tree.get_filtration()
        for _, filt in filtered_simplices:
            fvalues.append(filt)

        return list(set(fvalues))

    def split_test_set(self):
        test = self.dataset_handler.test

        m = random.randint(2, math.ceil(math.sqrt(len(test))))
        partitions = []
        elem = len(test)//m
        for i in range(m):
            partitions.append(test[i:i+elem])

        return partitions

    def add_partition(self, simplex_tree, X_i, V, e_j):
        V.extend([x for _, x in X_i])

        point_count = len(V)
        Q = {0: []}
        for i, x in X_i: # detect every point in V close enough to x \in Xm
            for id, v in enumerate(V):
                if v not in Q[0] and \
                        self.scg.predicate(x, [v], e_j):
                   Q[0].append([(id, False)])

        d = 0
        while d < self.max_dimension:
            d_next = d + 1
            Q.update({d_next: []})
            for i, val in enumerate(Q[d]):
                dsimplex, cof = val
                max_ind = max(dsimplex)  # we get the maximum index on the d-simplex

                for k in range(max_ind + 1, point_count):
                    v = V[k]
                    if self.scg.predicate([v], dsimplex, e_j):
                        Q[d_next].append((list(dsimplex).append(k), False))
                        if not cof:
                            cof = True

                if cof:
                    Q[d][i] = (dsimplex, cof)
            d = d_next

        for d in Q:
            for dsimplex, cof in Q[d]:
                if not cof:
                    self.simplex_tree.insert(dsimplex, e_j)
        self.simplex_tree.make_filtration_non_decreasing()

        del Q
        return simplex_tree, V

    def predicate(self, points):
        pass


class DirectSCBuilder:
    def __init__(self, data_handler, complex_type, data, **kwargs):
        self.dataset_handler = data_handler
        self.complex_type = complex_type
        self.data = data
        kwargs.update({"partitioning": kwargs.get("partitioning", False)})
        self.kwargs = kwargs

    def execute(self):
        if self.dataset_handler is None:
            return None

        complex = SimplicialComplexesGenerator(self.dataset_handler)

        Register.add_info_message(
            "Direct generation of Simplicial Complex type: {0} and dim {1}".format(self.complex_type,
                                                              self.complex_type.get_maximal_dimension()))
        simplex_tree = complex.generate_complex(sc_type=self.complex_type, data=self.data, **self.kwargs)

        return simplex_tree
