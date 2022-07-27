import numpy
import matplotlib.pyplot as plt
import scipy.linalg

def vcol(v):
    return v.reshape((v.size,1))

def compute_empirical_cov(D):
    mu = vcol(D.mean(1));
    DC = D-mu
    C = numpy.dot(DC,DC.T) / D.shape[1]
    return C 

def compute_sb(X,L): 
    SB = 0
    muD = vcol(X.mean(1))
    for i in [0,1]: 
        D = X[:,L==i]
        muC = vcol(D.mean(1)) 
        SB += D.shape[1] * numpy.dot((muC - muD), (muC - muD).T)
    return SB / X.shape[1]
        
def compute_sw(D,L): 
    SW = 0
    for i in [0,1]:
        SW+=  (L==i).sum() * compute_empirical_cov(D[:,L==i])
    return SW / D.shape[1]

"""Compute the LDA dimensionality reduction"""  
    
def LDA(DTR, LTR,DTE, m):
    
    SW = compute_sw(DTR,LTR)
    SB = compute_sb(DTR,LTR)
    s, U = scipy.linalg.eigh(SB,SW)
    W = U[:, ::-1][:, 0:m] #Column of W represents the main components (directions) that allow to preserve most of the information
    
    DW = numpy.dot(W.T, DTE)
    
    return DW
