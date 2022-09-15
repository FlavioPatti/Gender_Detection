import numpy
import matplotlib.pyplot as plt
import utilities as ut

"""Compute the PCA dimensionality reduction"""

def PCA(DTR,DTE,m): 
    mu =DTR.mean(1) 
    DC = DTR-ut.vcol(mu)
    C = numpy.dot(DC,DC.T) / DTR.shape[1]
    s, U = numpy.linalg.eigh(C) 
      
    P = U[:, ::-1][:, 0:m] #Column of P represents the main components (directions) that allow to preserve most of the information
    DP = numpy.dot(P.T, DTE) 
    return DP