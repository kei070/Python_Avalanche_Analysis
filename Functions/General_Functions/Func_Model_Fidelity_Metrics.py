#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function for computing the model fidelity metrics.
"""


def mod_metrics(con_mat):

    """
    Function for computing the model fidelity metrics.
    a = con_mat[0, 0]  # correct non-event
    b = con_mat[1, 0]  # miss
    c = con_mat[0, 1]  # false alarm
    d = con_mat[1, 1]  # hit
    """
    a = con_mat[0, 0]  # correct non-event
    b = con_mat[1, 0]  # miss
    c = con_mat[0, 1]  # false alarm
    d = con_mat[1, 1]  # hit

    # metrics as presented in Hendrikx et al. (2014) and see also Wilks (2011)
    rpc = 0.5 * (a / (a + c) + d / (b + d))  # unweighted average accuracy
    tss = d / (b + d) - c / (a + c)  # true skill score
    far = c / (c + d)  # false alarm ration
    pod = d / (b + d)  # probability of detection
    pon = a / (a + c)  # probability of non-events
    hss = 2 * (a*d - c*b) / ((a+b)*(b+d) + (a+c)*(c+d))  # Heidke Skill Score
    pss = (a*d - c*b) / ((a+b)*(c+d))  # Peirce Skill Score
    mfr = b / (a + b + c + d)  # fraction of misses

    output = {"RPC":rpc, "TSS":tss, "FAR":far, "POD":pod, "PON":pon, "HSS":hss, "PSS":pss, "MFR":mfr}

    return(output)
# end def
