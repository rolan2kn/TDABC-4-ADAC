#!/usr/bin/python
# -*- coding: utf-8 -*-

import os

import numpy as np

from dataset_tools.dataset_settings import DatasetSettingHandler
from utilities import utils
import time as time

from dataset_tools.dataset_builder import DatasetBuilder

from dataset_tools.dataset_splitter import DatasetSplitter
from utilities.register import Register

class DatasetHandler:
    def __init__(self, metadata = None):
        if metadata is None:
            metadata = DatasetSettingHandler()
            metadata.execute()
        self.metadata = metadata
        self.splitter = None
        self.tags_position = {}
        self.imbalanced_ratio = {}
        self.imbalanced_ratio1 = {}
        self.imbalanced_ratio2 = {}
        self.imbalanced_ratio3 = {}
        self.tags_set = set()

    def get_splitter(self):
        return self.splitter
    
    def load_dataset(self, *args):
        self.metadata = DatasetBuilder(self.metadata).execute(*args)

        self.splitter = self.metadata.splitter

        self.tags_position = {}
        self.imbalanced_ratio = {}
        self.tags_set = self.metadata.tags_set

        self.assign_tags()
        self.compute_imbalanced_ratio()

    def compute_imbalanced_ratio(self):
        self.metadata.imbalanced = self.splitter.is_imbalanced()

        occurrences = self.splitter.tags_ocurrences()

        size = self.splitter.dataset_size()
        olen = len(occurrences)
        for tag in occurrences:
            ratio = 1
            if occurrences[tag] != 0:
                occ_by_tag = (occurrences[tag])
                mult_occ_by_tag = (occurrences[tag]*olen)
                ratio1 = size/occ_by_tag
                ratio2 = size/mult_occ_by_tag
                ratio3 = 1-(occ_by_tag/size)

            self.imbalanced_ratio.update({tag: ratio2})
            self.imbalanced_ratio1.update({tag: ratio1})
            self.imbalanced_ratio2.update({tag: ratio2})
            self.imbalanced_ratio3.update({tag: ratio3})

        self.metadata.imbalanced_ratio = self.imbalanced_ratio


        return self.imbalanced_ratio

    def dataset_name(self):
        return self.metadata.dataset_name

    def configure_external_testing_set(self, ext_set, including_them=True):
        self.splitter.configure_external_testing_set(ext_set, including_them)

    def split_dataset(self, fold_size=None, fold_position=None):
        return self.splitter.execute(fold_size, fold_position)

    def in_tags_training(self, elem):
        return self.splitter.in_tags_training(elem)

    def get_tag_from_training(self, elem):
        return self.splitter.get_tag_from_training(elem)

    def get_tag_from_element(self, elem):
        l = self.get_tag_from_training(elem)
        if l is None:
            l = self.splitter.get_tag_from_test(elem)

        return l

    def assign_tags(self):
        for i, t in enumerate(self.metadata.tags_set):
            self.tags_position.update({t: i})
            self.imbalanced_ratio.update({t: 1})
            self.imbalanced_ratio1.update({t: 1})
            self.imbalanced_ratio2.update({t: 1})
            self.imbalanced_ratio3.update({t: 1})

    def unify_dataset(self, labelset = None, partition=False):
        return self.splitter.unify_dataset(labelset=labelset,
                                           partition=partition)

    def unify_tags(self, labelset = None, partition=False):
        return self.splitter.unify_tags(labelset=labelset,
                                           partition=partition)

    def get_training_info(self):
        return self.splitter.get_training_info()

    def get_test_info(self):
        return self.splitter.get_test_info()

    def labeled_points_number(self, ksimplex):
        return self.splitter.labeled_points_number(ksimplex)

    def get_training_label_sets(self):
        return self.splitter.get_training_label_sets()

    def get_training_point(self, ppos):
        return self.splitter.get_training_point(ppos)

    def get_test_point(self, ppos):
        return self.splitter.get_test_point(ppos)

    def get_tdabc_data(self):
        return self.splitter.get_tdabc_data()

    def get_pos_from_tag(self, tag):
        if tag not in self.tags_position:
            return -1

        return self.tags_position[tag]

    def get_label_from_pos(self, idx):
        for l in self.tags_position:
            if self.tags_position[l] == idx:
                return l

        return None

    def get_baseline_data(self):
        return self.splitter.get_baseline_data()

    def dataset_size(self):
        return self.splitter.dataset_size()

    def get_label_number(self):
        return len(self.tags_set)

    def get_suffix(self):
        return self.metadata.suffix

    def get_time(self):
        return self.metadata.time_suffix

    def set_path(self, new_path):
        self.metadata.path = new_path

    def get_folder(self):
        return self.metadata.folder

    def get_path(self):
        return self.metadata.path

    def get_current_name(self):
        return self.metadata.get_current_name()

    def get_real_dimension(self):
        return self.splitter.real_dimension()

    def cleanup(self):
        del self.tags_position
        del self.imbalanced_ratio
        del self.imbalanced_ratio1
        del self.imbalanced_ratio2
        del self.imbalanced_ratio3
        del self.tags_set

        del self.metadata
        del self.splitter

    def __del__(self):
        self.cleanup()


