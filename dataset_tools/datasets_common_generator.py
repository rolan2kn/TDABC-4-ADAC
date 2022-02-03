#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from utilities import utils
import os
import time
import random


class DatasetCommonGenerator:
    def __init__(self, *args, **kwargs):
        """
        __init__ constructor
        :param samples_number: num of dataset samples
        :param dimensions: the dataset extrinsic dimension
        """

        if len(args) == 0:
            kwargs = self.cure_attrib(**kwargs)
            self.samples_number = kwargs.get("samples_number", 100)
            self.dim = kwargs.get("dimensions", 3)
        else:
            values=[]
            for val in args:
                if type(val) is tuple:
                    for vval in val:
                        values.append(vval)
                else:
                    values.append(val)
            self.samples_number = int(values[0])
            self.dim = int(values[1])
            del values

        self.labels_number = 0
        self.can_i_translate = kwargs.get("can_i_translate", True)

        self.X = []
        self.label_set = []
        self.Y = []
        self.current_name = ""
        self.suffix = None

    def configure_sizes(self, no_objects, sizes=None):
        """
        This method take a num of object and size and make appropiate corrections
        :param no_objects: number of instances of data
        :param sizes: number of point samples of each object
        :return: a list of sizes (samples number) for any object
        """
        if sizes is None:
            sizes = [self.samples_number] * no_objects

        s = len(sizes)
        if s < no_objects:
            for j in range(s, no_objects):
                sizes.append(self.samples_number)

        return sizes

    def cure_attrib(self, **kwargs):

        for attrib in kwargs:
            data = kwargs[attrib]
            if type(data) == tuple:
                kwargs.update({attrib: data[0]})
        return kwargs

    def configure_arguments(self, **kwargs):
        pass

    def compute_labels(self, no_samples):
        """
        This method assign labels to each object
        :param no_samples: identifier of current object
        """
        self.label_set.append(self.labels_number)
        self.labels_number = len(self.label_set)

        y1 = [self.label_set[-1] for _ in range(no_samples)]
        self.Y.extend(y1)

    def compute_translation(self, no, scale, dim=None):
        """
        This method compute the desired translation of current object
        :param dim: dimension of point samples
        :param scale: the scale of datasets
        :return: the desired translation
        """
        if dim is None:
            dim = self.dim
        translate = np.array([0]*dim)

        if not self.can_i_translate:
            return translate

        i = 0
        e = random.randint(dim//2, dim)
        t = int(np.floor(np.sqrt(scale)))
        while i < e:
            p = random.randint(0, dim - 1)
            translate[p] += t

            i+=1

        return translate

    def draw(self, can_i_show=True):
        """
                Draws the current dataset.
                Each label has a representative color.
                This method also, save a *.png picture of the dataset
                :return:
                """
        n = len(self.X)

        if n == 0:
            return
        X = np.array(self.X)
        Y = np.array(self.Y)

        w, h = plt.figaspect(0.7)
        fig = plt.figure(figsize=(w, h))
        ax = fig.add_subplot(111, projection='3d')

        three_d = len(X[0]) > 2
        area = 80
        max_v = max(Y)
        if max_v == 0:
            max_v = 1
        for l in np.unique(Y):
            if three_d:
                ax.scatter(X[Y == l, 0], X[Y == l, 1], X[Y == l, 2],
                           color=plt.cm.jet(np.float(l) / max_v), s=area, edgecolor='k',
                           label=l)
            else:
                ax.scatter(X[Y == l, 0], X[Y == l, 1],
                           color=plt.cm.jet(np.float(l) / max_v), s=area, edgecolor='k',
                           label=l)
        lsize = 20
        ax.set_xlabel('X', size=lsize)
        ax.set_ylabel('Z', size=lsize)
        ax.set_zlabel('Y', size=lsize)

        ax.legend(fontsize=10, loc='best')
        plt.title(self.data.dataset_name(), size=2*lsize)

        if self.current_name is not None:
            plt.savefig("{0}.png".format(self.current_name))
        if can_i_show:
            plt.show()
        else:
            plt.close(plt.gcf())
            plt.close("all")

    def get_class(self):
        """
        Compute the class name from the current object
        :return: class name
        """
        module_name = str(self.__class__)
        classn = module_name.split(".")[1]
        if classn.find('\'') != -1:
            classn = classn[:classn.index('\'')]

        return classn

    def save_dataset(self, suffix=None):
        """
        Save the current dataset to file.
        The filename is generated by using the project path,
        current date, and a suffix with relevant information.
        :param suffix: relevant information about the dataset to build a filename
        """
        path = "{0}/datasets/".format(utils.get_module_path())
        if not os.path.isdir(path):
            os.makedirs(path)

        if suffix is None or suffix == "":
            suffix = "anonymous_data" if self.suffix is None else self.suffix

        self.current_name = time.strftime(
            "{0}%y.%m.%d_%H.%M.%S__{1}".format(path, suffix))

        self.suffix = suffix

        filename = "{0}.dat".format(self.current_name)

        data_file = open(filename, "w")
        data_file.write("type:{3},samples:{0},dim:{1},labels:{2}\n".format(self.samples_number,
                                                                    self.dim,
                                                                    self.labels_number,
                                                                    self.get_class()))
        for i in range(self.samples_number):
            line = "{0},{1}\n".format(self.X[i], self.Y[i])
            data_file.write(line)

        data_file.close()

    def load_header(self, header_line):
        """
        Parse the header line from a data file
        :param header_line: the header line to process
        """
        header_parts = header_line.split(",")

        self.samples_number = int(header_parts[1].split(":")[1])
        self.dim = int(header_parts[2].split(":")[1])
        self.labels_number = int(header_parts[3].split(":")[1])
        self.label_set = []

        for i in range(self.labels_number):
            self.label_set.append(i)

    def process_line(self, line):
        """
        :param line: a data fille line: '[0.8950860056829507, -0.19553213669168135, 0.40073460787823945],0\n'
        :return: a tuple (sample_point, label)
        """
        init = line.index("[")+1
        end = line.index("]")

        str_sample = line[init:end]
        sample_point = [float(x) for x in str_sample.split(",")]

        label = int(line[end+2:-1])

        return sample_point, label

    def load_dataset(self, filename):
        """
        Load a dataset from file.

        :param filename: the absolute path from the dataset file
        """
        if not os.path.exists(filename):
            raise Exception("ERROR!! the file \'{0}\' does not exists.".format(filename))

        self.current_name = filename[:-4]
        data_file = open(filename, "r")

        line = data_file.readline()
        self.load_header(line)

        self.X = []
        self.Y = []

        for line in data_file.readlines():
            sample, label = self.process_line(line)
            self.X.append(sample)
            self.Y.append(label)

        data_file.close()

