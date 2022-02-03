#!/usr/bin/python
# -*- coding: utf-8 -*-
import collections
import os
import random
import time as time

import numpy as np

from utilities.register import Register

"""
It split a dataset. This class is a helper to splitting data into training and test set. It also supports external 
test set which do not have a label, for performing prediction tasks.
"""
class DatasetSplitter:
    def __init__(self, **kwargs):
        self.dataset = kwargs.get("dataset", [])                    # dataset
        self.tags = kwargs.get("tags", [])                    # dataset
        self.fold_size = kwargs.get("fold_size", 0)
        self.fold_position = kwargs.get("fold_position", -1)
        self.external_test = None
        self.size = 0
        self.include_external_set = kwargs.get("include_external_set", True)

        if self.dataset is not None:
            self.size = len(self.dataset)

        self.training = []                                  # training set S
        self.test = []                                      # testing set  X
        self.tags_training = {}                             # association set T = {(s, l) | s \in S; l \in L}
        self.tags_test = {}                                 # incomplete association set T = {(x, l) | x \in X; l \in L}

    def destroy(self):
        self.clean()
        if self.dataset:
            self.dataset.clean()
        self.tags.clean()

    def dataset_size(self):
        return self.size

    def tags_ocurrences(self):
        if self.tags is None:
            return {}

        return collections.Counter(self.tags)

    def is_imbalanced(self):
        occ = self.tags_ocurrences()

        mi = min(occ.values())
        ma = max(occ.values())

        return mi != ma

    def configure_external_testing_set(self, ext_set, including_them=True):
        self.test.clear()
        self.tags_test.clear()

        tcount = self.size  # define the first element to classify

        size_2_clsfy = len(ext_set)
        for i in range(size_2_clsfy):  # we iterate the new testing set
            self.test.append([tcount+i, ext_set[i]])  # filling testing set
        self.include_external_set = including_them

    def clean(self):
        self.training.clear()
        self.test.clear()
        self.tags_training.clear()
        self.tags_test.clear()

    def execute(self, fold_size=None, fold_position=None):
        if self.dataset is None or self.size == 0:
            return False

        self.clean()
        self.fold_size = fold_size
        self.fold_position = fold_position
        self.external_test = self.fold_size in (None, 0)

        test_pos, training_pos = self.compute_test_and_training_positions()
        I = [i for i in range(self.size)]  # dataset-samples index list

        random.seed(0)  # make the index list distorted
        random.shuffle(I)
        random.shuffle(I)

        for i, tid in enumerate(training_pos):
            self.training.append(
                self.dataset[I[tid]])          # filling the training set
            self.tags_training.update(
                {str([i]): self.tags[I[tid]]})  # associating tags

        if not self.external_test:
            tcount = len(training_pos)-1
            for tid in test_pos:  # but if we are in the desired fold
                tcount += 1

                self.test.append([tcount, self.dataset[I[tid]]])  # filling testing set
                self.tags_test.update({str([tcount]): self.tags[I[tid]]})  # associating tags

        return True

    def compute_test_and_training_positions(self):
        if self.external_test:
            return [], [i for i in range(self.size)]

        tini_pos = (self.fold_position * self.fold_size)
        if tini_pos > self.size:
            tini_pos %= self.size

        test_position_list = [(i+tini_pos)%self.size
                              for i in range(self.fold_size)]

        training_position_list = []
        for i in range(self.size):
            if i not in test_position_list:
                training_position_list.append(i)

        return test_position_list, training_position_list

    def unify_dataset(self, labelset = None, partition=False):
        S = []
        if labelset is None or len(labelset) == 0:
            S.extend([list(x) for x in self.training])
        else:
            S.extend([list(x) for x in labelset])

        if not partition and \
                (not self.external_test or (self.external_test and self.include_external_set)):
            """
            If we are not considering a data partition, and there are not external points 
            or we are interested on those external points then we must including them
            """
            for _, x in self.test:
                S.append(list(x))

        if len(S) == 0:
            S = list([list(x) for x in self.dataset])

        return S

    def unify_tags(self, labelset = None, partition=False):
        if labelset is None or len(labelset) == 0:
            T = [self.tags_training[id] for id in self.tags_training]
        else:
            T = []
            T.extend(labelset)

        if not partition and \
                len(self.tags_test) == len(self.test) :
            """
            If we are not considering a data partition, and there 
            are real_tags for the test set then we include those tags
            """
            for id, _ in self.test:
                idx_key = str([id])
                T.append(self.tags_test[idx_key])

        if len(T) == 0:
            T = self.tags

        return T

    def get_training_info(self):
        if self.tags_training is None or self.training is None:
            return None, None

        ttags = [self.tags_training[id] for id in self.tags_training]
        tpoints = list(self.training)

        return tpoints, ttags

    def get_test_info(self):
        if self.test is None:
            return None, None

        ttags = [self.tags_test[id] for id in self.tags_test] if len(self.tags_test) == len(self.test) else []
        tpoints = [x for _, x in self.test]

        return tpoints, ttags

    def labeled_points_number(self, ksimplex):
        if len(self.tags_test) == 0:
            return {}, 0

        lpg = {l:0 for l in self.tags_set}  # a dict with the number
                                            # of point from each class inside the simplex
        nlp = 0
        if ksimplex is not None and 0 < len(ksimplex):
            for point in ksimplex:
                key = "[{0}]".format(point)
                if key in self.tags_training:
                    l = self.tags_training[key]
                    lpg[l] += 1
                    nlp += 1

        return lpg, nlp

    def get_training_label_sets(self):
        splitted_training = {}

        for idx, idx_key in enumerate(self.tags_training):
            l = self.tags_training[idx_key]
            vertex = self.training[idx]
            if l not in splitted_training:
                splitted_training.update({l: []})
            splitted_training[l].append(vertex)

        return splitted_training


    def get_training_point(self, ppos):
        if ppos > -1 and ppos < len(self.training):
            return self.training[ppos]

        return None

    def get_tdabc_data(self):
        ttest_tags = []
        ttests = []

        for idx, x0 in self.test:
            idx_key = str([idx])
            ttests.append(idx)

            if len(self.tags_test) > 0:
                ttest_tags.append(self.tags_test[idx_key])

        return ttests, ttest_tags

    def get_baseline_data(self):
        ttraining, ttags = self.get_training_info()

        ttest_tags = []
        ttests = []

        for idx, x0 in self.test:
            idx_key = str([idx])
            ttests.append(x0)

            if len(self.tags_test) > 0:
                ttest_tags.append(self.tags_test[idx_key])

        return ttraining, ttags, ttests, ttest_tags

    def build_item_key(self, elem):
        if not (type(elem) in (list, tuple, set)):
            item_key = str([elem])
        else:
            item_key = str(elem)

        return item_key

    def in_tags_training(self, elem):
        elem_key = self.build_item_key(elem)

        return elem_key in self.tags_training

    def get_tag_from_training(self, elem):
        elem_key = self.build_item_key(elem)

        if elem_key in self.tags_training:
            return self.tags_training[elem_key]

        return None

    def get_tag_from_test(self, elem):
        elem_key = self.build_item_key(elem)

        if elem_key in self.tags_test:
            return self.tags_test[elem_key]

        return None

    def get_test_point(self, elem):
        if type(elem) in (list, tuple):
            elem = elem[0]

        if not self.external_test:
            for i, x in self.test:
                if i == elem:
                    return x

        return None

    def real_dimension(self):
        if self.dataset is None or len(self.dataset) == 0:
            return 0

        dim = len(self.dataset[0])

        return dim

    def real_samples_by_label(self):
        occ = self.tags_ocurrences()

        return occ.values()

    def export_2_csv(self, csv_filename):
        real_dim = self.real_dimension()

        if real_dim == 0:
            Register.add_error_message("Bad dimensionality to export to csv")
            raise Exception("Bad dimensionality to export to csv")

        csv_file = open(csv_filename, "w")

        paths = "/".join(csv_filename.split("/")[:-1])
        if not os.path.isdir(paths):
            os.makedirs(paths)

        line = ""
        for i in range(real_dim):
            line+="x{0};".format(i)
        line += "label"
        csv_file.write("{0}\n".format(line))

        dsize = len(self.dataset)
        for i in range(dsize):
            line = ";".join([str(pi) for pi in self.dataset[i]])
            csv_file.write("{0};{1}\n".format(line, self.tags[i]))

        csv_file.close()