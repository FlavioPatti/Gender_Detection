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

def compute_sb(X,L): # muD è la media del dataset
    SB = 0
    muD = vcol(X.mean(1))
    for i in [0,1]: 
        D = X[:,L==i]
        muC = vcol(D.mean(1)) #muC è la media della classe 
        SB += D.shape[1] * numpy.dot((muC - muD), (muC - muD).T)
    return SB / X.shape[1]
        
def compute_sw(D,L): #within coviariance matrix
    SW = 0
    for i in [0,1]:
        SW+=  (L==i).sum() * compute_empirical_cov(D[:,L==i]) #calcolo la matrice di covarianza C come in PCA però per tutte le classi all'interno del dataset
        #(L==i).sum() = numero campioni per ogni classe = 50 
    return SW / D.shape[1]  
    
def LDA(D, L, m):
    
    SW = compute_sw(D,L)
    SB = compute_sb(D,L)
    s, U = scipy.linalg.eigh(SB,SW)
    W = U[:, ::-1][:, 0:m]
    
    DW = numpy.dot(W.T, D)
    
    return DW
