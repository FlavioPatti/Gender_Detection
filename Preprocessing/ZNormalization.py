import numpy
import matplotlib
import matplotlib.pyplot as plt
import utilities as ut

"""Compute the Znormalization of the dataset(centering and scaling to unit variance)"""

def ZNormalization(DTR,DTE):
    mean = DTR.mean(axis=1)
    standardDeviation = DTR.std(axis=1)
    ZD = (DTE-ut.mcol(mean))/ut.mcol(standardDeviation)
    return ZD