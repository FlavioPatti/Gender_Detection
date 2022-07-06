import numpy
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize
import sklearn
import sys
import sklearn.datasets
#Le soluzioni di logistic regression non possono essere calcolate in forma chiusa, 
#ricorreremo a numerical solvers (solutori numerici), un risolutore numerico cerca iterativamente il minimizzatore di una funzione

def mcol(v):
    return v.reshape((v.size,1))

def mrow(v):
    return v.reshape((1,v.size))

def logreg_obj_wrap(DTR, LTR, l):
    Z = LTR * 2.0 - 1.0
    M = DTR.shape[0]
    def logreg_obj(v): #v packs w b
        w = mcol(v[0:M])
        b = v[-1]
        S = numpy.dot(w.T, DTR) + b
        cxe = numpy.logaddexp(0, -S*Z).mean() #calcola log exp(0)=1 + log exp (-S*Z) per ogni elemento
        return cxe + 0.5*l*numpy.linalg.norm(w)**2
    return logreg_obj #non chiamo la funzione ma la torno soltanto, la chiamerà al suo interno fmin_l_bfgs_b


def QuadraticLogisticRegression(DTR, LTR, DTE, l): 
    
    #estensione dello spazio
    DTR_exp = []
    
    for i in range(DTR.shape[1]):
        x = DTR[:,i:i+1]
        tmp = numpy.dot(x, x.T)
        tmp2 = numpy.ravel(tmp, order='F')
        tmp3 = numpy.concatenate( (mcol(tmp2), mcol(x)), axis=0)
        DTR_exp.append(tmp3)
    DTR_exp=numpy.hstack(DTR_exp)
    
    DTE_exp = []
    
    for i in range(DTE.shape[1]):
        x = DTE[:,i:i+1]
        tmp = numpy.dot(x, x.T)
        tmp2 = numpy.ravel(tmp, order='F')
        tmp3 = numpy.concatenate( (mcol(tmp2), mcol(x)), axis=0)
        DTE_exp.append(tmp3)
    DTE_exp=numpy.hstack(DTE_exp)

    logreg_obj = logreg_obj_wrap(DTR_exp, LTR, l)
    
    _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR_exp.shape[0]+1), approx_grad=True)
    _w = _v[0:DTR_exp.shape[0]]
    _b = _v[-1]
    S = numpy.dot(_w.T, DTE_exp) + _b 

    return S

def balanced_logreg_obj_wrap(DTR, LTR, l,pi1):
    DTR1=DTR[:,LTR==1]
    DTR0=DTR[:,LTR==0]
    M = DTR.shape[0]
    def logreg_obj(v): #v packs w b
        w = mcol(v[0:M])
        b = v[-1]
        S1 = numpy.dot(w.T, DTR1) + b
        cxe1 = numpy.logaddexp(0, -S1).mean() #calcola log exp(0)=1 + log exp (-S*Z) per ogni elemento
        S0 = numpy.dot(w.T, DTR0) + b
        cxe0 = numpy.logaddexp(0, -(S0*-1)).mean() #calcola log exp(0)=1 + log exp (-S*Z) per ogni elemento
        return pi1*cxe1+(1-pi1)*cxe0 + 0.5*l*numpy.linalg.norm(w)**2
    return logreg_obj #non chiamo la funzione ma la torno soltanto, la chiamerà al suo interno fmin_l_bfgs_b
        
def BalancedQuadraticLogisticRegression(DTR, LTR, DTE, l,pi1): 

    #estensione dello spazio
    DTR_exp = []

    for i in range(DTR.shape[1]):
        x = DTR[:,i:i+1]
        tmp = numpy.dot(x, x.T)
        tmp2 = numpy.ravel(tmp, order='F')
        tmp3 = numpy.concatenate( (mcol(tmp2), mcol(x)), axis=0)
        DTR_exp.append(tmp3)
    DTR_exp=numpy.hstack(DTR_exp)

    DTE_exp = []

    for i in range(DTE.shape[1]):
        x = DTE[:,i:i+1]
        tmp = numpy.dot(x, x.T)
        tmp2 = numpy.ravel(tmp, order='F')
        tmp3 = numpy.concatenate( (mcol(tmp2), mcol(x)), axis=0)
        DTE_exp.append(tmp3)
    DTE_exp=numpy.hstack(DTE_exp)

    logreg_obj = balanced_logreg_obj_wrap(DTR_exp, LTR, l,pi1)

    _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR_exp.shape[0]+1), approx_grad=True)
    _w = _v[0:DTR_exp.shape[0]]
    _b = _v[-1]
    S = numpy.dot(_w.T, DTE_exp) + _b 

    return S

