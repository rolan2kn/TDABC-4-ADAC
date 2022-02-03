#!/usr/bin/python
# -*- coding: utf-8 -*-

import math

import numpy as np

from utilities import utils
from utilities.register import Register

class MetricTypeHandler:
    ACC, PREC, REC, TNR, FNR, DP, FPR, FPDR, F1, FB, BACC, \
    CLSFE, MCC, GMEAN, SAMPLES, AUC, AVP = range(17)

    def __init__(self):
        self.metric_list = []

    def get_param_list(self):
        return [MetricTypeHandler.ACC, MetricTypeHandler.PREC, MetricTypeHandler.REC, MetricTypeHandler.TNR,
                #MetricTypeHandler.FNR,
                #MetricTypeHandler.DP,
                MetricTypeHandler.FPR, MetricTypeHandler.FPDR, MetricTypeHandler.F1, MetricTypeHandler.FB,
                MetricTypeHandler.BACC, MetricTypeHandler.CLSFE, MetricTypeHandler.MCC, MetricTypeHandler.GMEAN,
                MetricTypeHandler.SAMPLES, MetricTypeHandler.AUC, MetricTypeHandler.AVP]

    def get_all_names(self):
        metric_types = self.get_param_list()
        mnames = []

        for mtype in metric_types:
            mnames.append(self.param_2_str(mtype))

        return mnames

    def is_ACC(self, mtype):
        return mtype == MetricTypeHandler.ACC

    def is_PREC(self, mtype):
        return mtype == MetricTypeHandler.PREC

    def is_REC(self, mtype):
        return mtype == MetricTypeHandler.REC

    def is_TNR(self, mtype):
        return mtype == MetricTypeHandler.TNR

    def is_FNR(self, mtype):
        return mtype == MetricTypeHandler.FNR

    def is_DP(self, mtype):
        return mtype == MetricTypeHandler.DP

    def is_FPR(self, mtype):
        return mtype == MetricTypeHandler.FPR

    def is_FPDR(self, mtype):
        return mtype == MetricTypeHandler.FPDR

    def is_F1(self, mtype):
        return mtype == MetricTypeHandler.F1

    def is_FB(self, mtype):
        return mtype == MetricTypeHandler.FB

    def is_BACC(self, mtype):
        return mtype == MetricTypeHandler.BACC

    def is_CLSFE(self, mtype):
        return mtype == MetricTypeHandler.CLSFE

    def is_MCC(self, mtype):
        return mtype == MetricTypeHandler.MCC

    def is_GMEAN(self, mtype):
        return mtype == MetricTypeHandler.GMEAN

    def is_SAMPLES(self, mtype):
        return mtype == MetricTypeHandler.SAMPLES

    def is_AUC(self, mtype):
        return mtype == MetricTypeHandler.AUC

    def is_AVP(self, mtype):
        return mtype == MetricTypeHandler.AVP

    def str_2_param(self, param_name):
        param_type = None
        #ACC, PREC, REC, TNR, FPR, FPDR, F1, BACC, CLSFE, MCC, GMEAN
        if param_name == "ACC":
           param_type = MetricTypeHandler.ACC
        elif param_name == "PREC":
            param_type = MetricTypeHandler.PREC
        elif param_name == "REC":
            param_type = MetricTypeHandler.REC
        elif param_name == "TNR":
           param_type = MetricTypeHandler.TNR
        elif param_name == "FNR":
           param_type = MetricTypeHandler.FNR
        elif param_name == "DP":
           param_type = MetricTypeHandler.DP
        elif param_name == "FPR":
           param_type = MetricTypeHandler.FPR
        elif param_name == "FPDR":
           param_type = MetricTypeHandler.FPDR
        elif param_name == "F1":
            param_type = MetricTypeHandler.F1
        elif param_name.find("FB") != -1:
            param_type = MetricTypeHandler.FB
        elif param_name == "BACC":
            param_type = MetricTypeHandler.BACC
        elif param_name == "CLSFE":
            param_type = MetricTypeHandler.CLSFE
        elif param_name == "MCC":
            param_type = MetricTypeHandler.MCC
        elif param_name == "GMEAN":
            param_type = MetricTypeHandler.GMEAN
        elif param_name == "SAMPLES":
            param_type = MetricTypeHandler.SAMPLES
        elif param_name == "AUC":
            param_type = MetricTypeHandler.AUC
        elif param_name == "AVP" or param_name == "AVGP":
            param_type = MetricTypeHandler.AVP

        return param_type

    def param_2_str(self, param_type):
        param_name = None
        if self.is_ACC(param_type):
           param_name = 'ACC'
        elif self.is_PREC(param_type):
            param_name = 'PREC'
        elif self.is_REC(param_type):
            param_name = 'REC'
        elif self.is_TNR(param_type):
           param_name = 'TNR'
        elif self.is_FNR(param_type):
           param_name = 'FNR'
        elif self.is_DP(param_type):
           param_name = 'DP'
        elif self.is_FPR(param_type):
           param_name = 'FPR'
        elif self.is_FPDR(param_type):
           param_name = 'FPDR'
        elif self.is_F1(param_type):
            param_name = 'F1'
        elif self.is_FB(param_type):
            param_name = 'FB'
        elif self.is_BACC(param_type):
            param_name = 'BACC'
        elif self.is_CLSFE(param_type):
            param_name = 'CLSFE'
        elif self.is_MCC(param_type):
            param_name = 'MCC'
        elif self.is_GMEAN(param_type):
            param_name = 'GMEAN'
        elif self.is_SAMPLES(param_type):
            param_name = 'SAMPLES'
        elif self.is_AUC(param_type):
            param_name = 'AUC'
        elif self.is_AVP(param_type):
            param_name = 'AVP'

        return param_name

    def param_2_fullname(self, param_type):
        param_name = None

        if self.is_ACC(param_type):
           param_name = "ACCURACY"
        elif self.is_PREC(param_type):
            param_name = "PRECISION"
        elif self.is_REC(param_type):
            param_name = "RECALL"
        elif self.is_TNR(param_type):
           param_name = "TRUE_NEGATIVE_RATE"
        elif self.is_FNR(param_type):
           param_name = "FALSE NEGATIVE RATE"
        elif param_name == "DP":
           param_name = "DISCRIMINATIVE_POWER"
        elif self.is_FPR(param_type):
           param_name = "FALSE_POSITIVE_RATE"
        elif self.is_FPDR(param_type):
           param_name = "FALSE_POSITIVE_DISCOVERY_RATE"
        elif self.is_F1(param_type):
            param_name = "F1-MEASURE"
        elif self.is_FB(param_type):
            param_name = "FB-MEASURE"
        elif self.is_BACC(param_type):
            param_name = "BALANCED_ACCURACY"
        elif self.is_CLSFE(param_type):
            param_name = "CLASSIFICATION_ERROR"
        elif self.is_MCC(param_type):
            param_name = "MATTEWS_CORRELATION_COEFICIENT"
        elif self.is_GMEAN(param_type):
            param_name = "GEOMETRIC_MEAN"
        elif self.is_SAMPLES(param_type):
            param_name = "SAMPLES"
        elif self.is_AUC(param_type):
            param_name = "ROC-AUC"
        elif self.is_AVP(param_type):
            param_name = 'PR-AUC'

        return param_name


class Metrics:

    def __init__(self, **kwargs):
        self.mtype_handler = MetricTypeHandler()
        self.init_values(**kwargs)

    def init_values(self, **kwargs):
        self._accuracy = kwargs.get("ACC", 0)
        self._precision = kwargs.get("PREC", 0)
        self._recall = kwargs.get("REC", 0)
        self._tn_rate = kwargs.get("TNR", 0)
        self._fn_rate = kwargs.get("FNR", 0)
        self._dp = kwargs.get("DP", 0)
        self._fp_rate = kwargs.get("FPR", 0)
        self._fp_discovery_rate = kwargs.get("FPDR", 0)
        self._f1_measure = kwargs.get("F1", 0)
        self._fb_measure = kwargs.get("FB", None)
        self._balanced_accuracy = kwargs.get("BACC", 0)
        self._clsf_error = kwargs.get("CLSFE", 0)
        self._mcc = kwargs.get("MCC", 0)
        self._gmean = kwargs.get("GMEAN", 0)
        self._samples = kwargs.get("SAMPLES", 0)
        self._auc = {}
        self._avg_precision = {}

    def set_auc_avg_prec(self, auc, avg_p):
        self._auc = self.parse_value(auc)
        self._avg_precision = self.parse_value(avg_p)

    def compute_metrics(self, **kwargs):
        try:
            self.init_values(**kwargs)

            TP = kwargs.get("TP", None)
            FP = kwargs.get("FP", None)
            TN = kwargs.get("TN", None)
            FN = kwargs.get("FN", None)
            self._samples = kwargs.get("SAMPLES", None)
            clsf_error = kwargs.get("clsf_error", None)

            if TP is None or    \
                FP is None or   \
                TN is None or  \
                FN is None or \
                clsf_error is None or \
                self._samples is None:
                return

            total = TP + TN + FP + FN
            self._accuracy = 0
            if total != 0:
                self._accuracy = (TP + TN) / total

            tp_fp = TP + FP
            tp_fn = TP + FN
            tn_fp = TN + FP

            if tp_fp != 0:
                self._precision = TP / tp_fp

            if tp_fn != 0:          # True positive rate or Sensitivity
                self._recall = TP / tp_fn

            if tn_fp != 0:            # True Negative Rate or Specificity
                self._tn_rate = TN / tn_fp

            if tp_fn != 0:            # False Negative Rate or Specificity
                self._fn_rate = FN / tp_fn

            X = self._recall / (1-self._recall) if self._recall != 1 else 0
            Y = self._tn_rate / (1-self._tn_rate) if self._tn_rate != 1 else 0

            log10x = np.log10(X) if X != 0 else 0
            log10y = np.log10(Y) if Y != 0 else 0

            self._dp = (np.sqrt(3)/np.pi)*(log10x + log10y)

            if tn_fp != 0:            # False Positive rate
                self._fp_rate = FP / tn_fp

            if tp_fp != 0:
                self._fp_discovery_rate = FP / tp_fp

            _2tp = 2 * TP
            _2tp_fp_fn = _2tp + FP + FN
            self._fb_measure = {}
            if _2tp_fp_fn != 0:
                self._f1_measure = _2tp / _2tp_fp_fn

                self._fb_measure = {i: ((1+i**2)*TP) / ((i**2)*(TP+FN) + TP + FP) for i in [0.5, 2, 3]}
            tn_fn = TN+FN
            div = tp_fp * tp_fn * tn_fp * tn_fn
            if div > 0:
               div = math.sqrt(div)

               self._mcc = (TP*TN - FP*FN) / div

            if self._tn_rate * self._recall > 0:
                self._gmean = math.sqrt(self._tn_rate * self._recall)

            self._clsf_error = clsf_error
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def accuracy(self):
        return self._accuracy

    def precision(self):
        return self._precision

    def recall(self):
        return self._recall

    def false_positive_rate(self):
        return self._fp_rate

    def true_negative_rate(self):
        return self._tn_rate

    def false_negative_rate(self):
        return self._fn_rate

    def discriminative_power(self):
        return self._dp

    def false_positive_discovery_rate(self):
        return self._fp_discovery_rate

    def f1_measure(self):
        return self._f1_measure

    def fb_measure(self):
        if self._fb_measure is None:
            Pr = self.precision()
            Re = self.recall()

            if Pr != 0 or Re != 0:
                self._fb_measure = {i: ((1 + i ** 2) * Pr * Re) / ((i ** 2) * Pr + Re) for i in [0.5, 2, 3]}
            else:
                self._fb_measure = {i: 0.0 for i in [0.5, 2, 3]}
        return self._fb_measure

    def classification_error(self):
        if self._samples == 0:
            return 1

        return self._clsf_error / self._samples

    def clsf_error(self):
        return self._clsf_error

    def balanced_accuracy(self):
        return self._balanced_accuracy

    def mattews_corr_coef(self):
        return self._mcc

    def geometric_mean(self):
        return self._gmean

    def samples(self):
        return self._samples

    def get_mtype_handler(self):
        return self.mtype_handler

    def parse_value(self, value):
        try:
            if type(value) == str:
                pos_i = value.find("{")
                pos_f = value.find("}")

                if pos_i == -1 or pos_f == -1:
                    return float(value)

                if pos_f-pos_i < 2:
                    return 0

                line = value[pos_i+1:pos_f]

                dict_result = {}

                items = line.split(",")
                for elem in items:
                    key, value = elem.split(":")
                    if utils.is_float_str(key.strip()):
                        key = float(key)

                    dict_result.update({key: float(value)})
                return dict_result

            return float(value)
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def set_param(self, mtype, value):
        if self.mtype_handler.is_ACC(mtype):
            self._accuracy = value
        elif self.mtype_handler.is_PREC(mtype):
            self._precision = value
        elif self.mtype_handler.is_REC(mtype):
            self._recall = value
        elif self.mtype_handler.is_TNR(mtype):
            self._tn_rate = value
        elif self.mtype_handler.is_FPR(mtype):
            self._fp_rate = value
        elif self.mtype_handler.is_FPDR(mtype):
            self._fp_discovery_rate = value
        elif self.mtype_handler.is_F1(mtype):
            self._f1_measure = value
        elif self.mtype_handler.is_FB(mtype):
            self._fb_measure = value
        elif self.mtype_handler.is_BACC(mtype):
            self._balanced_accuracy = value
        elif self.mtype_handler.is_CLSFE(mtype):
            self._clsf_error = value
        elif self.mtype_handler.is_MCC(mtype):
            self._mcc = value
        elif self.mtype_handler.is_GMEAN(mtype):
            self._gmean = value
        elif self.mtype_handler.is_SAMPLES(mtype):
            self._samples = value
        elif self.mtype_handler.is_AUC(mtype):
            self._auc = value
        elif self.mtype_handler.is_AVP(mtype):
            self._avg_precision = value

    def get_param(self, mtype):
        value = self._accuracy

        if self.mtype_handler.is_ACC(mtype):
            value = self._accuracy
        elif self.mtype_handler.is_PREC(mtype):
            value = self._precision
        elif self.mtype_handler.is_REC(mtype):
            value = self._recall
        elif self.mtype_handler.is_TNR(mtype):
            value = self._tn_rate
        elif self.mtype_handler.is_FPR(mtype):
            value = self._fp_rate
        elif self.mtype_handler.is_FPDR(mtype):
            value = self._fp_discovery_rate
        elif self.mtype_handler.is_F1(mtype):
            value = self._f1_measure
        elif self.mtype_handler.is_FB(mtype):
            value = self.fb_measure()
        elif self.mtype_handler.is_BACC(mtype):
            value = self._balanced_accuracy
        elif self.mtype_handler.is_CLSFE(mtype):
            value = self._clsf_error
        elif self.mtype_handler.is_MCC(mtype):
            value = self._mcc
        elif self.mtype_handler.is_GMEAN(mtype):
            value = self._gmean
        elif self.mtype_handler.is_SAMPLES(mtype):
            value = self._samples
        elif self.mtype_handler.is_AUC(mtype):
            value = self._auc
        elif self.mtype_handler.is_AVP(mtype):
            value = self._avg_precision

        return value

    def to_file(self, fmetrics):
        try:
            # ACC, PREC, REC, TNR, FPR, FPDR, F1, BACC, CLSFE, MCC, GMEAN
            fmetrics.write("ACC:{0}\n".format(self.accuracy()))
            fmetrics.write("PREC:{0}\n".format(self.precision()))
            fmetrics.write("REC:{0}\n".format(self.recall()))
            fmetrics.write("TNR:{0}\n".format(self.true_negative_rate()))
            fmetrics.write("FNR:{0}\n".format(self.false_negative_rate()))
            fmetrics.write("DP:{0}\n".format(self.discriminative_power()))
            fmetrics.write("FPR:{0}\n".format(self.false_positive_discovery_rate()))
            fmetrics.write("FPDR:{0}\n".format(self.false_positive_discovery_rate()))
            fmetrics.write("F1:{0}\n".format(self.f1_measure()))
            fmetrics.write("FB:{0}\n".format(self.fb_measure()))
            fmetrics.write("BACC:{0}\n".format(self.balanced_accuracy()))
            fmetrics.write("CLSFE:{0}\n".format(self.classification_error()))
            fmetrics.write("MCC:{0}\n".format(self.mattews_corr_coef()))
            fmetrics.write("GMEAN:{0}\n".format(self.geometric_mean()))
            fmetrics.write("SAMPLES:{0}\n".format(self.samples()))
            fmetrics.write("AUC:{0}\n".format(self._auc))
            fmetrics.write("AVGP:{0}\n".format(self._avg_precision))
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def from_file(self, fmetrics):
        try:
            self.init_values()
            param_list = self.mtype_handler.get_param_list()
            for _ in param_list:
                line = fmetrics.readline()
                parts = line.strip().split(":")
                if len(parts) > 1:
                    param_type = self.mtype_handler.str_2_param(parts[0])

                    param_value = parts[1]
                    if len(parts) > 2:
                        param_value = ":".join(parts[1:])

                    if param_value not in (None, "\n", " ", ""):
                        self.set_param(param_type, self.parse_value(param_value.strip()))
                else:
                    break
            self._clsf_error *= self._samples # because it was divided before save it
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def __str__(self):
        try:
            type_list = self.mtype_handler.get_param_list()
            metric_2_str = "<<<--------------------->>>\n"

            for i, mtype in enumerate(type_list):
                param = self.get_param(mtype)
                if self.mtype_handler.is_CLSFE(mtype):
                    param = self.classification_error()
                metric_2_str += "{0}:{1}\n".format(self.mtype_handler.param_2_fullname(mtype), param)

            del type_list

            return metric_2_str
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def to_dict(self):
        try:
            type_list = self.mtype_handler.get_param_list()

            metric_dict = {}
            for i, mtype in enumerate(type_list):
                param = self.get_param(mtype)
                if self.mtype_handler.is_CLSFE(mtype):
                    param = self.classification_error()

                metric_dict.update({self.mtype_handler.param_2_str(mtype): param})
                if self.mtype_handler.is_FB(mtype):
                    for beta in param:
                        metric_dict.update({"{0}-{1}".format(self.mtype_handler.param_2_str(mtype), float(beta)): param[beta]})

            return metric_dict
        except Exception as e:
            Register.add_error_message(e)
            raise e

class OverallMetrics(Metrics):
    def __init__(self, classes=None):
        super(OverallMetrics, self).__init__()
        self.classes = classes if classes is not None else []
        self.metrics_by_class = {}
        for c in self.classes:
            self.metrics_by_class.update({c: Metrics()})

    def get_metrics_from_class(self, label):
        if label in self.metrics_by_class:
            return self.metrics_by_class[label]

        return None

    def compute_metrics(self, **kwargs):
        try:
            self.init_values(**kwargs)

            self.accuracy()
            self.precision()
            self.recall()
            self.true_negative_rate()
            self.false_positive_rate()
            self.false_positive_discovery_rate()
            self.f1_measure()
            self.fb_measure()
            self.balanced_accuracy()
            self.clsf_error()
            self.mattews_corr_coef()
            self.geometric_mean()
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def compute(self, param):
        try:
            sum = 0
            n = len(self.classes)
            if n == 0:
                return n, 1
            if self.mtype_handler.is_CLSFE(param):
                n = 0

            if self.mtype_handler.is_SAMPLES(param):
                n = 1
            elif self.mtype_handler.is_FB(param):
                sum = {}

            for label in self.metrics_by_class:
                metric = self.metrics_by_class[label]

                if self.mtype_handler.is_FB(param):
                    fb = metric.fb_measure()
                    for b in fb:
                        if b not in sum:
                            sum.update({b: 0})
                        sum[b] += fb[b]
                elif self.mtype_handler.is_CLSFE(param):
                    sum += metric.clsf_error()
                    n += metric.samples()
                else:
                    sum += metric.get_param(param)

            return sum, n
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def accuracy(self):
        if self._accuracy in (None, 0):
            acc, n = self.compute(MetricTypeHandler.ACC)
            self._accuracy = acc / n

        return self._accuracy

    def precision(self):
        if self._precision in (None, 0):
            prec, n = self.compute(MetricTypeHandler.PREC)
            self._precision = prec / n

        return self._precision

    def recall(self):
        if self._recall in (None, 0):
            rec, n = self.compute(MetricTypeHandler.REC)
            self._recall = rec / n

        return self._recall

    def balanced_accuracy(self):
        if self._balanced_accuracy in (None, 0):
            bacc, n = self.compute(MetricTypeHandler.BACC)
            self._balanced_accuracy = bacc / n

        return self._balanced_accuracy

    def set_auc_avg_prec(self, auc, avg_p):
        try:
            auc_key = self.mtype_handler.param_2_str(MetricTypeHandler.AUC)
            avp_key = self.mtype_handler.param_2_str(MetricTypeHandler.AVP)

            nauc = {}
            navg_p = {}
            if auc_key in auc:
                nauc = auc[auc_key]
            if avp_key in avg_p:
                navg_p = avg_p[avp_key]

            for label in self.metrics_by_class:
                metrics = self.metrics_by_class[label]
                if label in nauc and label in navg_p:
                    lauc = nauc[label]
                    lavp = navg_p[label]
                    metrics.set_auc_avg_prec(lauc, lavp)
            macro = nauc['macro'] if "macro" in nauc else 0
            micro = navg_p['micro'] if "micro" in navg_p else 0
            super(OverallMetrics, self).set_auc_avg_prec(macro, micro)
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def f1_measure(self):
        if self._f1_measure in (None, 0):
            f1, n = self.compute(MetricTypeHandler.F1)
            self._f1_measure = f1 / n

        return self._f1_measure

    def fb_measure(self):
        if self._fb_measure in (None, 0):
            fb, n = self.compute(MetricTypeHandler.FB)
            self._fb_measure = {b: fb[b] / n for b in fb}

        return self._fb_measure

    def classification_error(self):
        cerror, n = self.compute(MetricTypeHandler.CLSFE)
        self._clsf_error = cerror

        return self._clsf_error / n

    def clsf_error(self):
        if self._clsf_error in (None, 0):
            cerror, _ = self.compute(MetricTypeHandler.CLSFE)
            self._clsf_error = cerror

        return self._clsf_error

    def false_positive_rate(self):
        if self._fp_rate in (None, 0):
            fpr, n = self.compute(MetricTypeHandler.FPR)
            self._fp_rate = fpr / n

        return self._fp_rate

    def true_negative_rate(self):
        if self._tn_rate in (None, 0):
            tnr, n = self.compute(MetricTypeHandler.TNR)
            self._tn_rate = tnr / n

        return self._tn_rate

    def false_positive_discovery_rate(self):
        if self._fp_discovery_rate in (None, 0):
            fpdr, n = self.compute(MetricTypeHandler.FPDR)
            self._fp_discovery_rate = fpdr / n

        return self._fp_discovery_rate

    def mattews_corr_coef(self):
        if self._mcc in (None, 0):
            mcc, n = self.compute(MetricTypeHandler.MCC)
            self._mcc = mcc / n

        return self._mcc

    def geometric_mean(self):
        if self._gmean in (None, 0):
            gm, n = self.compute(MetricTypeHandler.GMEAN)
            self._gmean = gm / n

        return self._gmean

    def samples(self):
        if self._samples in (None, 0):
            s, n = self.compute(MetricTypeHandler.SAMPLES)

            self._samples = s / n

        return self._samples

    def to_file(self, filename):
        try:
            if not isinstance(filename, str):
                return

            fmetrics = open(filename, "w")

            fmetrics.write("\n\nGENERAL_METRICS\n")
            super(OverallMetrics, self).to_file(fmetrics) # call Metrics.to_file(fmetrics) method

            fmetrics.write("\n\nMETRICS_PER_CLASSES\n")
            for l in self.classes:
                fmetrics.write("\n======>> CLASS:{0}\n".format(l))
                self.metrics_by_class[l].to_file(fmetrics)

            fmetrics.flush()
            fmetrics.close()
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def from_file(self, filename):
        try:
            if not isinstance(filename, str):
                return

            fmetrics = open(filename, "r")

            line = "\n"
            while line == "\n":     # ignore empty lines
                line = fmetrics.readline()

            if line.strip() == "GENERAL_METRICS":
                super(OverallMetrics, self).from_file(fmetrics)  # call Metrics.from_file(fmetrics) method

            line = "\n"
            while line == "\n":  # ignore empty lines
                line = fmetrics.readline()

            if self.metrics_by_class is not None:
                self.metrics_by_class.clear()
                del self.metrics_by_class

            self.metrics_by_class = {}
            if line.strip() == "METRICS_PER_CLASSES":
                while not line == "": #EOF
                    line = fmetrics.readline()

                    if line.find("CLASS:") != -1:
                        part = line.split(":")
                        if len(part) > 1:
                            l = part[1].strip()
                            self.metrics_by_class.update({l: Metrics()})
                            self.metrics_by_class[l].from_file(fmetrics)

            self.classes = list(self.metrics_by_class.keys())
            fmetrics.close()
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def get_minority_class(self):
        '''
        This method returns the minority class if there is a minority class (an imbalanced dataset)
        if not, its returns None because there is not a ninority class

        :return: the minority class or None
        '''
        try:
            minority_class = None

            classes = list(self.metrics_by_class.keys())
            if len(classes) == 0:
                return None

            minor = self.metrics_by_class[classes[0]].samples()
            for cls in self.metrics_by_class:
                sample = self.metrics_by_class[cls].samples()
                if sample < minor:
                    minor = sample
                    minority_class = cls

            return minority_class
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def get_dict_from_minority_class(self):
        try:
            mc = self.get_minority_class()

            if mc is None:
                return {}

            metrics = self.metrics_by_class[mc]

            return metrics.to_dict()
        except Exception as e:
            Register.add_error_message(e)
            raise e

    def __getitem__(self, item):
        return self.get_metrics_from_class(item)