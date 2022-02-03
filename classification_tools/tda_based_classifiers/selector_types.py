#!/usr/bin/python
# -*- coding: utf-8 -*-

class SelectorTypeHandler:
    AVERAGE, MAXIMAL, RANDOMIZED, IQR, HAVERAGE, GAVERAGE, \
    WAVERAGE, WHAVERAGE, WGAVERAGE, OUTSIDE = [1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1 << 7, 1 << 8,
                                               1 << 9]

    def __init__(self, type=None):
        if type is None:
            type = SelectorTypeHandler.RANDOMIZED | SelectorTypeHandler.MAXIMAL | SelectorTypeHandler.AVERAGE
        self.selector_types = []
        self.modify_selectors(type)

    def get_selector_collection(self):
        return [SelectorTypeHandler.AVERAGE, SelectorTypeHandler.MAXIMAL, SelectorTypeHandler.RANDOMIZED,
                SelectorTypeHandler.IQR, SelectorTypeHandler.HAVERAGE, SelectorTypeHandler.GAVERAGE,
                SelectorTypeHandler.OUTSIDE]

    def modify_selectors(self, classifier_type):
        scollection = self.get_selector_collection()
        for i in scollection:
            if classifier_type & i != 0:
                self.selector_types.append(i)

    def get_selector(self, i):
        if i < 0 or i > len(self.selector_types):
            i = 0
            
        return self.selector_types[i]

    def to_str(self, ctype):
        if ctype == SelectorTypeHandler.RANDOMIZED:
            return "RANDOMIZED"
        elif ctype == SelectorTypeHandler.MAXIMAL:
            return "MAXIMAL"
        elif ctype == SelectorTypeHandler.IQR:
            return "IQR"
        elif ctype == SelectorTypeHandler.HAVERAGE:
            return "HAVERAGE"
        elif ctype == SelectorTypeHandler.GAVERAGE:
            return "GAVERAGE"
        elif ctype == SelectorTypeHandler.OUTSIDE:
            return "OUTSIDE"

        return "AVERAGE"

    def first_letter(self, ctype):
        name = self.to_str(ctype)

        if len(name) == 0:
            return ""

        return name[0]

    def all_to_str(self):
        return [self.to_str(s) for s in self.selector_types]

    def is_selector_present(self, choice):
        return choice in self.selector_types

    def selectors(self):
        return self.selector_types

    def is_Randomized(self, choice):
        return SelectorTypeHandler.RANDOMIZED in self.selector_types and choice == SelectorTypeHandler.RANDOMIZED

    def is_Maximal(self, choice):
        return SelectorTypeHandler.MAXIMAL in self.selector_types and choice == SelectorTypeHandler.MAXIMAL

    def is_Average(self, choice):
        return SelectorTypeHandler.AVERAGE in self.selector_types and choice == SelectorTypeHandler.AVERAGE
    
    def is_HAverage(self, choice):
        return SelectorTypeHandler.HAVERAGE in self.selector_types and choice == SelectorTypeHandler.HAVERAGE

    def is_GAverage(self, choice):
        return SelectorTypeHandler.GAVERAGE in self.selector_types and choice == SelectorTypeHandler.GAVERAGE

    def is_IQR(self, choice):
        return SelectorTypeHandler.IQR in self.selector_types and choice == SelectorTypeHandler.IQR

    def is_Outside(self, choice):
        return SelectorTypeHandler.OUTSIDE in self.selector_types and choice == SelectorTypeHandler.OUTSIDE