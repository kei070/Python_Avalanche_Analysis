#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# imports
import numpy as np

def custom_dt(data, thres1, thres2, feat1, feat2):
    """
    Apply a customary decision tree with a predefined structure to the data.
    The structure is as follows: The data must contain two features (provide the feature names in parameters feat1 and
    feat2; the order is important!). The splitting rules will be applied to these features as follows:
        For every instance where feature 1 surpasses the threshold of rule1 the value 1 is assigned.
        For every other instance the value 0 is assigned, EXCEPT those where feature 2 surpasses the threshold 2, for
        which 1 is assigned.
    """

    risk = np.zeros(len(data))

    risk[data[feat1] > thres1] = 1
    risk[((data[feat1] <= thres1) & (data[feat2] > thres2))] = 1

    return risk.astype(int)
# end def