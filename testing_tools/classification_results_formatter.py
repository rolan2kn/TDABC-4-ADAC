import heapq
import os.path
import time

import numpy as np
import matplotlib

from utilities.register import Register

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import plotly.graph_objs as go
from plotly.subplots import make_subplots

from classification_tools.classifier_types import ClassifierTypeHandler
from classification_tools.evaluation_tools.metrics import OverallMetrics, MetricTypeHandler
from dataset_tools.dataset_types import DatasetTypeHandler
from utilities import utils



class ClassificationResultsFormatter:
    BALANCED, IMBALANCED = range(2)

    def __init__(self, **kwargs):
        self.root_path = kwargs.get("root_path", "./docs/CLASSIFICATION_KRNN/EXPERIMENTS/DATASET_EXPERIMENT/")

        self.full_data = {}
        self.minor_full_data = {}
        self.mtype_handler = MetricTypeHandler()
        self.dataser_order = {}
        self.desired_metrics = kwargs.get("desired_metrics", self.mtype_handler.get_all_names())
        self.desired_dimension = kwargs.get("desired_dimension", "3d")
        self.skip_datasets = kwargs.get("skip_datasets", ["KRNN"])
        self.skip_classifiers = kwargs.get("skip_classifiers", ["RBF-SVM"])
        self.plotter_engine = kwargs.get("graph_mode", 1)
        # self.plotter_engine = kwargs.get("graph_mode", 0)
        self.csv_output = kwargs.get("csv_output", True)
        self.result_dir = None

    def execute(self):
        try:
            all_filenames = utils.get_all_filenames(self.root_path, ".txt")
            self.root_path_length = len(self.root_path)

            self.dataset_names = DatasetTypeHandler().get_supported_dataset_names()
            self.num_times = []
            for filename in all_filenames:
                params = self.prepare_params(filename)

                if params is None:
                    continue
                self.process_params(params)

                print(filename)
            if self.csv_output:
                self.write_global_results()
            self.plot_global_results()
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def prepare_params(self, filename):
        try:
            pos = filename.find(self.root_path)
            if pos == -1:
                return None

            params = {}
            partial_filename = filename[pos + self.root_path_length:]
            parts = partial_filename.split("/")
            parts_length = len(parts)

            if parts_length < 3 or partial_filename.find("_metrics.txt") == -1:
                return None

            txt_file_name = parts[-1]
            method_name = parts[-2]
            params.update({"txt_file_name": filename})

            txt_file_name = txt_file_name.replace("_metrics.txt", "")
            timestamp = parts[-4]
            self.mode = 2 if timestamp not in self.dataset_names else 3

            txt_parts = txt_file_name.split("-")
            classifier_name = txt_parts[0]
            complex_name = txt_parts[1]
            dimension = txt_parts[2]
            if dimension != self.desired_dimension:
                return None

            fs = txt_parts[3]
            if method_name.find("-") != -1:
                classifier_name = "{0}-{1}".format(classifier_name, txt_parts[1])
                complex_name = txt_parts[2]
                dimension = txt_parts[3]
                fs = txt_parts[4]

            if self.mode == 2:  # krnn experiment
                dataset_name = parts[-3]
                timestamp = parts[-4]
                index = dataset_name.split("_")[-1]
            else:
                dataset_name = parts[-4]
                timestamp = parts[-3]
                index = dataset_name

            ctype_hdlr = ClassifierTypeHandler()
            classifier_name = ctype_hdlr.validate_name(classifier_name)
            for classifier in self.skip_classifiers:            # Ignoring undesired classifiers
                if classifier_name.find(classifier) != -1:
                    return None

            if ctype_hdlr.is_tdabc_based(ctype_hdlr.from_str(classifier_name)):
                label = "{0}-{1}".format(classifier_name[1:], method_name)
            else:
                method_name = ctype_hdlr.validate_name(method_name)
                label = method_name

            params.update({"dataset_name": dataset_name,
                           "timestamp": timestamp,
                           "classifier_name": classifier_name,
                           "complex_name": complex_name,
                           "dimension": dimension,
                           "fs": fs,
                           "label": label,
                           "instance": index
                           })

            return params
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def is_it_a_desired_metric(self, param_name):
        param_type = self.mtype_handler.str_2_param(param_name)

        if param_type is None:
            return False

        new_param_name = self.mtype_handler.param_2_str(param_type) # we filter/normalize a transformed metric
        if new_param_name in self.desired_metrics:  # it is a desired metrid
            if not self.mtype_handler.is_FB(param_type) or \
                    param_name != new_param_name:   # and it is not FB or it is FB but FB-{0.5,2, 3}
                return True

        return False

    def process_params(self, params):
        try:
            txt_file_name = params["txt_file_name"]
            dim = params["dimension"]
            fs = params["fs"]
            label = params["label"]
            instance = params["instance"]
            timestamp = params['timestamp']
            dataset_name = params["dataset_name"]

            if timestamp not in self.num_times:
                self.num_times.append(timestamp)

            if dim not in self.full_data:
                self.full_data.update({dim: {}})
                self.minor_full_data.update({dim: {}})
            if fs not in self.full_data[dim]:
                self.full_data[dim].update({fs: {}})
                self.minor_full_data[dim].update({fs: {}})

            overall_metric = OverallMetrics()
            overall_metric.from_file(filename=txt_file_name)

            graphics_dicts = overall_metric.to_dict()
            minority_class_dict = overall_metric.get_dict_from_minority_class()
            balance_key = ClassificationResultsFormatter.BALANCED
            if len(minority_class_dict) == 0:
                minority_class_dict = graphics_dicts
                if balance_key not in self.dataser_order:
                    self.dataser_order.update({ClassificationResultsFormatter.BALANCED: []})

            else:
                balance_key = ClassificationResultsFormatter.IMBALANCED

            if balance_key not in self.dataser_order:
                self.dataser_order.update({balance_key: []})

            if dataset_name not in self.dataser_order[balance_key]:
                self.dataser_order[balance_key].append(dataset_name)

            for param_name in graphics_dicts:
                if not self.is_it_a_desired_metric(param_name):
                    continue

                param_value = graphics_dicts[param_name]
                if type(param_value) == dict:
                    param_value = param_value[list(param_value.keys())[0]]
                mc_param_value = minority_class_dict[param_name]
                if type(mc_param_value) == dict:
                    mc_param_value = mc_param_value[list(mc_param_value.keys())[0]]

                if param_name not in self.full_data[dim][fs]:
                    self.full_data[dim][fs].update({param_name: {}})
                    self.minor_full_data[dim][fs].update({param_name: {}})
                if label not in self.full_data[dim][fs][param_name]:
                    self.full_data[dim][fs][param_name].update({label: {}})
                    self.minor_full_data[dim][fs][param_name].update({label: {}})
                if instance not in self.full_data[dim][fs][param_name][label]:
                    self.full_data[dim][fs][param_name][label].update({instance: []})
                    self.minor_full_data[dim][fs][param_name][label].update({instance: []})
                self.full_data[dim][fs][param_name][label][instance].append(param_value)
                self.minor_full_data[dim][fs][param_name][label][instance].append(mc_param_value)
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def sorted_classifier_list(self):
        try:
            if self.full_data is None or len(self.full_data) == 0:
                raise Exception("the data collection is empty, you must collect result data again!!")

            key_list = list(self.full_data.keys())
            key = key_list[0]

            params = list(self.full_data[key]['NORMAL'].keys())
            classifiers = list(self.full_data[key]['NORMAL'][params[0]].keys())

            priority_queue = []
            for classifier in classifiers:
                priority = 100
                if classifier is None:
                    continue
                if classifier.find("TDABC") != -1:
                    priority = 0
                elif classifier.find("KNN") != -1:
                    priority = 1
                elif classifier.find("SVM") != -1:
                    priority = 2
                elif classifier.find("TF") != -1:
                    priority = 3
                heapq.heappush(priority_queue, (priority, classifier))

            sorted_classifiers = []
            num_clsf = len(priority_queue)
            for i in range(num_clsf):
                pr, current_item = heapq.heappop(priority_queue)
                sorted_classifiers.append(current_item)

            del classifiers
            del priority_queue

            return sorted_classifiers
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def sorted_ticks(self):
        try:
            if self.full_data is None or len(self.full_data) == 0:
                raise Exception("the data collection is empty, you must collect result data again!!")

            key_list = list(self.full_data.keys())
            dim_key = key_list[0]

            params = list(self.full_data[dim_key]['NORMAL'].keys())
            classifiers = list(self.full_data[dim_key]['NORMAL'][params[0]].keys())

            param_key = params[0]
            clsf_key = classifiers[0]
            ticks_key = list(self.full_data[dim_key]["NORMAL"][param_key][clsf_key].keys())

            tick_labels = []
            if ticks_key[0].isdigit():

                for key in self.full_data[dim_key]["NORMAL"][param_key][clsf_key]:
                    if key.isdigit():
                        tick_labels.append(int(key))
                tick_labels = sorted(tick_labels)
            else:
                tick_labels.extend(sorted(self.dataser_order[ClassificationResultsFormatter.IMBALANCED]))
                tick_labels.extend(sorted(self.dataser_order[ClassificationResultsFormatter.BALANCED]))

            del key_list
            del params
            del classifiers

            return tick_labels

        except Exception as e:
            Register.add_error_message(e)
            raise  e

    def plot_global_results(self):
        if self.result_dir is None:
            self.result_dir = "{0}/GLOBAL_RESULTS/{1}/".format(self.root_path, time.strftime("%y.%m.%d_%H.%M.%S"))
        result_dir = self.result_dir
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)

        classifier_list = self.sorted_classifier_list()
        data_elements = [self.full_data, self.minor_full_data]
        for i,data in enumerate(data_elements):
            for dim in data:
                for fs in data[dim]:
                    for param_name in data[dim][fs]:
                        if self.mode != 2: #krnn_experiment
                            self.plot_bars_matplotlib(dim, fs, param_name, classifier_list, result_dir, data_number=(i==0))
                        else:
                            if self.plotter_engine == 1:
                                self.plot_errorbars_matplotlib(dim, fs, param_name, classifier_list, result_dir, data_number=(i==0))
                            else:
                                self.plot_errorbars_plotly(dim, fs, param_name, classifier_list, result_dir, data_number=(i==0))

    def write_global_results(self):
        try:
            if self.result_dir is None:
                self.result_dir = "{0}/GLOBAL_RESULTS/{1}/".format(self.root_path, time.strftime("%y.%m.%d_%H.%M.%S"))
            result_dir = self.result_dir
            if not os.path.isdir(result_dir):
                os.makedirs(result_dir)

            classifier_list = self.sorted_classifier_list()
            tick_labels = self.sorted_ticks()
            for dim in self.full_data:
                for fs in self.full_data[dim]:
                    csv_name = "{0}_metrics_{1}_{2}.csv".format(result_dir, dim, fs)
                    csv_file = open(csv_name, "w")
                    csv_file.write("\n")
                    for param_name in self.full_data[dim][fs]:
                        averages = {}
                        for idx, clsf in enumerate(classifier_list):
                            if idx == 0:
                                headers = ""
                                line = ""
                                data_type = ""
                                for l in tick_labels:
                                    headers += ";;"
                                    line += "{0};;".format(l)
                                    data_type += ";GLOBAL;Minority class"
                                    averages.update({str(l):[[],[]]})
                                csv_file.write("{0}{1};\n".format(param_name, headers))
                                csv_file.write(";{0}\n".format(line))
                                csv_file.write("{0}{1}\n".format("CLASSIFIERS", data_type))

                            line = "{0};".format(clsf)
                            trials = []
                            for label in tick_labels:
                                label = str(label)
                                trial = self.full_data[dim][fs][param_name][clsf][label]
                                trial2 = self.minor_full_data[dim][fs][param_name][clsf][label]
                                line = "{0}{1};{2};".format(line, trial[0], trial2[0])

                                averages[label][0].append(trial[0])
                                averages[label][1].append(trial2[0])


                            line = "{0}\n".format(line)
                            csv_file.write(line)
                        line = "{0};".format("AVERAGE")
                        for label in averages:
                            tavg_global = np.average(averages[label][0])
                            tavg_min = np.average(averages[label][1])
                            line = "{0}{1};{2};".format(line, tavg_global, tavg_min)
                        csv_file.write("{0}\n;{1}\n".format(line, headers))

                    csv_file.close()
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def plot_bars_matplotlib(self, dim, fs, param_name, classifier_list, result_dir, data_number):
        try:
            fig, ax = plt.subplots(1, 1, figsize=(7, 8))
            param_type = self.mtype_handler.str_2_param(param_name)
            fig_name = self.mtype_handler.param_2_fullname(param_type) if self.mtype_handler.is_AUC(param_type) or \
                                                                          self.mtype_handler.is_AVP(param_type)  \
                                                                        else param_name
            plt.title(fig_name, fontsize=20)
            data = self.full_data if data_number else self.minor_full_data

            max_v = len(data[dim][fs][param_name])
            min_v = 1
            tick_labels = self.sorted_ticks()
            size_ticks = len(tick_labels)
            padding = 1
            total_width = 6

            X = [0]
            for i in range(1, size_ticks):
                space = len(str(tick_labels[i-1]))*0.5+len(str(tick_labels[i]))*0.5

                X.append(X[i-1] + max(space, total_width)+padding)
            X = np.array(X)

            width = total_width/len(classifier_list)
            half_width = width*(len(classifier_list)*0.5)

            for idx, clsf in enumerate(classifier_list):
                Y = []
                E = []
                for label in tick_labels:
                    label = str(label)
                    trials = data[dim][fs][param_name][clsf][label]
                    tavg = np.average(trials)
                    tsdv = np.std(trials)

                    Y.append(tavg)
                    E.append(tsdv)

                min_v = min(min(Y), min_v)
                Y = np.array(Y)
                E = np.array(E)

                lower_error = 0.4 * E
                upper_error = E

                asymmetric_error = [lower_error, upper_error]

                plt.bar(X-half_width + width*idx, Y, yerr=asymmetric_error, width=width, color=plt.cm.jet(np.float(idx) / max_v),
                            linewidth=2, label=clsf)

            ax.set_ylabel(param_name, fontsize=20)
            plt.ylim([min_v - 0.01, 1.01])
            ax.set_xlabel("", labelpad=10)
            ax.set_xticks(X)
            ax.set_xticklabels(tick_labels, fontsize=15, rotation=-30)
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='lower center')
            plt.tight_layout()

            suffix = "global" if data_number else "minority"
            name = "{0}{1}_{2}_{3}_{4}.png".format(result_dir, param_name, dim, fs, suffix)

            plt.savefig(name)
            plt.close(plt.gcf())
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def plot_errorbars_plotly(self, dim, fs, param_name, classifier_list, result_dir, data_number):

        fig = make_subplots(rows=1, cols=1)
        data = self.full_data if data_number else self.minor_full_data

        max_v = len(data[dim][fs][param_name])
        filled_markers = ('o', '*', 'v', '^', '<', '>', '8', 's', 'p', 'h', 'H', 'D', 'd', 'P', 'X')
        min_v = 1
        colors = colors =  ["darkblue", "blue", "mediumblue", "darkgreen", "green", "yellowgreen", "orange", "red"]
        tick_labels = self.sorted_ticks()
        for idx, clsf in enumerate(classifier_list):
            Y = []
            E = []

            for label in tick_labels:
                label = str(label)
                trials = data[dim][fs][param_name][clsf][label]
                tavg = np.average(trials)
                tsdv = np.std(trials)

                Y.append(tavg)
                E.append(tsdv)

            X = np.array(range(len(Y)))
            min_v = min(min(Y), min_v)
            Y = np.array(Y)
            E = np.array(E)

            lower_error = Y - 0.4 * E
            upper_error = Y + 0.4 * E

            fig.add_trace(
                 go.Scatter(
                    name=clsf,
                    x=X,
                    y=Y,
                    mode='lines',
                     marker=dict(color=colors[idx]),
                    line=dict(color=colors[idx], width=2)
                ))
            fig.add_trace(go.Scatter(
                    name='Upper Bound',
                    x=X,
                    y=upper_error,
                    mode='none',
                    marker=dict(color=colors[idx]),
                    line=dict(width=0),
                    fill="tonexty",
                    #opacity=0.3,
                    showlegend=False
                ))
            fig.add_trace(go.Scatter(
                    name='Lower Bound',
                    x=X,
                    y=lower_error,
                    marker=dict(color=colors[idx]),
                    line=dict(width=0),
                    mode='none',
                    fill='tonexty',
                    opacity= 0.3,
                    showlegend=False
                ))

        fig.update_layout(
            yaxis_title=param_name,
            title=param_name,
            hovermode="x"
        )

        suffix = "global" if data_number else "minority"
        name = "{0}{1}_{2}_{3}_{4}.png".format(result_dir, param_name, dim, fs, suffix)

        fig.write_image(name)

    def plot_errorbars_matplotlib(self, dim, fs, param_name, classifier_list, result_dir, data_number):

        fig, ax = plt.subplots(1, 1, figsize=(7, 8))
        param_type = self.mtype_handler.str_2_param(param_name)
        fig_name = self.mtype_handler.param_2_fullname(param_type) if (self.mtype_handler.is_AUC(param_type) or \
                                                                          self.mtype_handler.is_AVP(param_type) ) \
                                                                        else param_name
        plt.title(fig_name, fontsize=20)
        data = self.full_data if data_number else self.minor_full_data

        max_v = len(data[dim][fs][param_name])
        filled_markers = ('o', '*', 'v', '^', '<', '>', '8', 's', 'p', 'h', 'H', 'D', 'd', 'P', 'X')
        min_v = 1
        tick_labels = self.sorted_ticks()
        # X = range(len(tick_labels))

        tlabels = [tick_labels[0]]

        size_ticks = len(tick_labels)
        padding = 1
        total_width = 3

        X = [0]
        for i in range(1, size_ticks):
            space = len(str(tick_labels[i - 1])) * 0.5 + len(str(tick_labels[i])) * 0.5
            tlabels.append(tick_labels[i] if i % 2 == 0 else "")
            X.append(X[i - 1] + max(space, total_width) + padding)
        X = np.array(X)

        for idx, clsf in enumerate(classifier_list):
            Y = []
            E = []
            for label in tick_labels:
                label = str(label)
                trials = data[dim][fs][param_name][clsf][label]
                tavg = np.average(trials)
                tsdv = np.std(trials)

                Y.append(tavg)
                E.append(tsdv)

            min_v = min(min(Y), min_v)
            Y = np.array(Y)

            E = np.array(E)

            lower_error = 0.4 * E
            upper_error = E
            asymmetric_error = [lower_error, upper_error]

            ax.errorbar(X, Y, yerr=asymmetric_error, color=plt.cm.jet(np.float(idx) / max_v),
                        linewidth=2, label=clsf, marker=filled_markers[idx])

        ax.set_ylabel(param_name, fontsize=20)
        plt.ylim([min_v - 0.01, 1])
        ax.set_xlabel("")
        ax.set_xticks(X)
        ax.set_xticklabels(tlabels, fontsize=20, rotation=-30)
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='lower center')
        plt.tight_layout()

        suffix = "global" if data_number else "minority"
        name = "{0}{1}_{2}_{3}_{4}.png".format(result_dir, param_name, dim, fs, suffix)

        plt.savefig(name)
        plt.close(plt.gcf())
