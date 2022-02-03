#!/usr/bin/python
# -*- coding: utf-8 -*-
from classification_tools.tda_based_classifiers.selector_types import SelectorTypeHandler
from utilities.register import Register


class ClassifierTypeHandler:
    TDABC, PTDABC, KNN, WKNN, SVM, LSVM, RF = [1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6]

    def __init__(self, type=None):
        if type is None:
            type = ClassifierTypeHandler.PTDABC | ClassifierTypeHandler.KNN | ClassifierTypeHandler.WKNN | \
                   ClassifierTypeHandler.SVM | ClassifierTypeHandler.LSVM | ClassifierTypeHandler.RF
        self.classifier_types = []
        self.modify_classifiers(type)

    def modify_classifiers(self, classifier_type):
        for i in [ClassifierTypeHandler.TDABC, ClassifierTypeHandler.PTDABC, ClassifierTypeHandler.KNN,
                  ClassifierTypeHandler.WKNN, ClassifierTypeHandler.SVM, ClassifierTypeHandler.LSVM,
                  ClassifierTypeHandler.RF]:
            if classifier_type & i == i:
                self.classifier_types.append(i)
            self._add_type_verification_method(i)

        self._add_type_based_method(ClassifierTypeHandler.TDABC)
        self._add_type_based_method(ClassifierTypeHandler.SVM)
        self._add_type_based_method(ClassifierTypeHandler.KNN)
        self._add_type_based_method(ClassifierTypeHandler.RF)

    def to_str(self, ctype):
        if ctype == ClassifierTypeHandler.TDABC:
            return "TDABC"
        elif ctype == ClassifierTypeHandler.PTDABC:
            return "PTDABC"
        elif ctype == ClassifierTypeHandler.KNN:
            return "KNN"
        elif ctype == ClassifierTypeHandler.WKNN:
            return "WKNN"
        elif ctype == ClassifierTypeHandler.RF:
            return "RF"
        elif ctype == ClassifierTypeHandler.SVM:
            return "SVM"
        elif ctype == ClassifierTypeHandler.LSVM:
            return "LSVM"

        return None

    def validate_name(self, str_ctype):
        try:
            ctype = self.from_str(str_ctype)

            return self.to_str(ctype)
        except Exception as e:
            raise e

    def from_str(self, str_ctype):
        if str_ctype == "PTDABC":
            return ClassifierTypeHandler.PTDABC
        elif str_ctype == "KNN":
            return ClassifierTypeHandler.KNN
        elif str_ctype == "WKNN":
            return ClassifierTypeHandler.WKNN
        elif str_ctype == "RF":
            return ClassifierTypeHandler.RF
        elif str_ctype in ("SVM", "RBF_SVM", "RBF-SVM", "RBFSVM", "RSVM"):
            return ClassifierTypeHandler.SVM
        elif str_ctype == "LSVM":
            return ClassifierTypeHandler.LSVM

        return ""

    def is_baseline(self, ctype):
        return not self.is_tdabc_based(ctype)

    def get_baselines(self):
        baselines = []
        for ctype in self.classifier_types:
            if self.is_baseline(ctype):
                baselines.append(ctype)

        return baselines

    def get_tdabc(self):
        tdabc_list = []
        for ctype in self.classifier_types:
            if self.is_tdabc_based(ctype):
                tdabc_list.append(ctype)

        return tdabc_list

    def _add_type_verification_method(self, ctype):
        '''
        This method generates dynamic methods with the signature: def is_<ctype>(self, arg) and the
        method is added to the ClassifierTypeHandler class. The generated method checks if an argument type is a
        ctype classifier.

        For example:
        If we want to generate a new method called def is_tdabc(self, arg) we must to invoke
        self.add_type_verification_method(TDABC). Then we can check if arg is a  TDABC classifier.

        Obs: This method is not intended to be called by the user
        because all required dynamic methods are created automatically. The user only requires to invoque
        resulting dynamic methods.

        :param ctype: a fixed classifier type.
        :return: yes or not an argument type is a ctype classifier.
        '''
        name = self.to_str(ctype).lower()

        def inner_type_verifier(self, arg):
            return arg == ctype

        inner_type_verifier.__doc__ = "Verify if the classifier is a {0} classifier".format(self.to_str(ctype))
        inner_type_verifier.__name__ = "is_{0}".format(name)
        setattr(self.__class__, inner_type_verifier.__name__, inner_type_verifier)

    def _add_type_based_method(self, ctype):
        '''
        This method generates dynamic methods with the signature: def is_<ctype>_based(self, arg) and the method is
        added to the ClassifierTypeHandler class. The generated method checks if an argument type is a variant of ctype.
        We validate that if the name of ctype is contained in teh argument name.

        For example:
        If we want to generate a new method called def is_tdabc_based(self, arg) we must to invoke
        self.add_type_based_method(TDABC). Then we can check if arg is a variant of TDABC.

        Obs: This method is not intended to be called by the user
        because all required dynamic methods are created automatically. The user only requires to invoque
        resulting dynamic methods.

        :param ctype: a fixed classifier type.
        :return: yes or not an argument type is a variant of ctype
        '''
        ctype_name = self.to_str(ctype)

        def inner_type_based_method(self, arg):
            arg_name = self.to_str(arg)

            return arg_name.find(ctype_name) != -1

        inner_type_based_method.__doc__ = "Verify if the classifier is a {0} classifier".format(ctype)
        inner_type_based_method.__name__ = "is_{0}_based".format(ctype_name.lower())
        setattr(self.__class__, inner_type_based_method.__name__, inner_type_based_method)

    def get_selector_handlers(self):
        return SelectorTypeHandler()
