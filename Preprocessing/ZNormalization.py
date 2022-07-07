import numpy
import matplotlib
import matplotlib.pyplot as plt

def mcol(v):
    return v.reshape((v.size,1))

def mrow(v):
    return v.reshape((1,v.size))

def ZNormalization(D):
    mean = D.mean(axis=1)
    standardDeviation = D.std(axis=1)
    ZD = (D-mcol(mean))/mcol(standardDeviation)
    return ZD