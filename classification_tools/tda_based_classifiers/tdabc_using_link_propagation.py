#!/usr/bin/python
# -*- coding: utf-8 -*-

from classification_tools.tda_based_classifiers.tdabc import TDABasedClassifier

from classification_tools.tda_based_classifiers.link_based_propagation import LinkBasedPropagation


class TDABasedClassifierUsingLinkPropagation(TDABasedClassifier):
    def __init__(self, **kwargs):
        super(TDABasedClassifierUsingLinkPropagation, self).__init__(**kwargs)

    """
    This method executes a link-based propagation in case we reach the second border case 
    during the labeling stage  
    """
    def post_processing_upwards(self, **kwargs):
        sigma = kwargs.get("sigma", None)
        link = kwargs.get("link", None)
        fvalues = kwargs.get("fvalues", None)

        return LinkBasedPropagation(self).execute(sigma, link=link, fvalues=fvalues)
