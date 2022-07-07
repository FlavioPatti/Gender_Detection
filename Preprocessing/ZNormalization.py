import numpy
import matplotlib
import matplotlib.pyplot as plt

def mcol(v):
    return v.reshape((v.size,1))

def mrow(v):
    return v.reshape((1,v.size))

def ZNormalization(DTR,DTE):
    mean = DTR.mean(axis=1)
    standardDeviation = DTR.std(axis=1)
    ZD = (DTE-mcol(mean))/mcol(standardDeviation)
    return ZD