import numpy
import matplotlib.pyplot as plt

def vcol(v):
    return v.reshape((v.size,1))

def PCA(D,L,m): 
        
    mu =D.mean(1) 
    DC = D-vcol(mu)
    C = numpy.dot(DC,DC.T) / D.shape[1]
    s, U = numpy.linalg.eigh(C) 
      
    P = U[:, ::-1][:, 0:m] #P rappresenta la componenti (direzioni) principali che mi permettono di preservare la maggior parte delle informazioni 
    DP = numpy.dot(P.T, D) #proiezione dei punti nelle direzioni principali
    return DP