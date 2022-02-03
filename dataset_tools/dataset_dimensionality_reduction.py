#!/usr/bin/python
# -*- coding: utf-8 -*-
import time

import numpy as np
import sklearn
UMAP_PRESENT = True
try:
    import umap
    # from geomstats.learning.pca import TangentPCA
except Exception as e:
    UMAP_PRESENT = False


class DatasetDimensionalityReduction:
    PCA, TSNE, UMAP = range(3)

    @staticmethod
    def pca(X, components=3):
        pca = sklearn.decomposition.PCA(components)  # Instanciamos PCA
        X = pca.fit_transform(X)  # Ajustamos el análisis

        return X

    @staticmethod
    def tangent_pca(X, components=3):
        pass

    @staticmethod
    def tsne(X, components=3, **kwargs):
        '''
        It is highly recommended to use another dimensionality reduction method
        (e.g. PCA for dense data or TruncatedSVD for sparse data) to reduce the
        number of dimensions to a reasonable amount (e.g. 50) if the number of
        features is very high. This will suppress some noise and speed up the
        computation of pairwise distances between samples.

        :param X:
        :param components:
        :return:
        '''

        start = time.time()
        pca_cmp = min(50, len(X[0])-2)
        random_state = kwargs.get("random_state", 42)
        verbose = kwargs.get("verbose", 0)
        perplexity = kwargs.get("perplexity", 42)
        n_iter = kwargs.get("n_iter", 400)

        pca_50 = sklearn.decomposition.PCA(n_components=pca_cmp)
        pca_result_50 = pca_50.fit_transform(X)
        X = sklearn.manifold.TSNE(random_state=random_state,
                                       n_components=components,
                                       verbose=verbose,
                                       perplexity=perplexity,
                                       n_iter=n_iter).\
            fit_transform(pca_result_50)

        return X

    @staticmethod
    def umap(X, components=3, **kwargs):
        '''
        UMAP has different hyperparameters that can have an impact on the resulting embeddings:

            n_neighbors

        This parameter controls how UMAP balances local versus global structure in the data.
        This low values of n_neighbours forces UMAP to focus on very local structures while
        the higher values will make UMAP focus on the larger neighbourhoods.

            min_dist

        This parameter controls how tightly UMAP is allowed to pack points together. Lower values
        mean the points will be clustered closely and vice versa.

            n_components

        This parameter allows the user to determine the dimensionality of the reduced dimension space.

            metric

        This parameter controls how distance is computed in the ambient space of the input data.

        to install it please use: conda install -c conda-forge umap-learn or pip install umap-learn

        :param X:
        :param components:
        :return:
        '''
        if not UMAP_PRESENT:
            return DatasetDimensionalityReduction.tsne(X, components=components, **kwargs)

        n_neigh = kwargs.get("neighbors", 5)
        mindist = kwargs.get("mindist", 0.3)
        metric = kwargs.get("metric", 'correlation')

        X = umap.UMAP(n_neighbors=n_neigh, n_components=components,
                  min_dist=mindist,
                  metric=metric, random_state=39887).fit_transform(X) # random_state to provide stability under
                # repeated executions, useful for simplicial complex plotter
        return X

    @staticmethod
    def execute(X, **kwargs):
        rtype = kwargs.get("rtype", DatasetDimensionalityReduction.UMAP)

        components = 3
        if "components" in kwargs:
            components = int(kwargs.get("components", 3))
            del kwargs["components"]


        if rtype == DatasetDimensionalityReduction.PCA:
            return DatasetDimensionalityReduction.pca(X, components)

        elif rtype == DatasetDimensionalityReduction.TSNE:
            return DatasetDimensionalityReduction.tsne(X, components, **kwargs)

        return DatasetDimensionalityReduction.umap(X, components, **kwargs)
