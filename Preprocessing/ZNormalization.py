import numpy
import matplotlib
import matplotlib.pyplot as plt
import utilities as ut

"""Implementation of the Znormalization of the dataset(centering data and scaling to unit variance)"""

def ZNormalization(DTR,DTE):
    mean = DTR.mean(axis=1)
    standardDeviation = DTR.std(axis=1)
    ZD = (DTE-ut.vcol(mean))/ut.vcol(standardDeviation)
    return ZD