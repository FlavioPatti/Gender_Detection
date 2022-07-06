import numpy
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize
import sklearn
import sys
import sklearn.datasets
#Le soluzioni di logistic regression non possono essere calcolate in forma chiusa, 
#ricorreremo a numerical solvers (solutori numerici), un risolutore numerico cerca iterativamente il minimizzatore di una funzione
#in questo laboratorio utilizzeremo l'algoritmo L-BFGS che richiede almeno 2 argomenti:
# func: la funzione che vogliamo minimizzare.
# x0: il valore iniziale per l'algoritmo. [0, 0]
# Lascia che l'implementazione calcoli un gradiente approssimato: pass approx_grad = True.
#restituisce una tupla con tre valori x, f, d:
# x è la posizione stimata del minimo
# f è il valore obiettivo al minimo
# d contiene informazioni aggiuntive (consultare la documentazione)

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

def LinearLogisticRegression(DTR,LTR, DTE, lamb):
        logreg_obj = logreg_obj_wrap(DTR, LTR, lamb)
        _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0]+1), approx_grad=True)
        _w = _v[0:DTR.shape[0]]
        _b = _v[-1]
        S = numpy.dot(_w.T, DTE) + _b

        return S

def pri_wei_logreg_obj_wrap(DTR, LTR, l, pi1):
    DTR1=DTR[LTR==1]
    DTR0=DTR[LTR==0]
    def logreg_obj(v): #v packs w b
        w = v[0]
        b = v[-1]
        S1 = numpy.dot(w.T, DTR1) + b
        cxe1 = numpy.logaddexp(0, -S1).mean() #calcola log exp(0)=1 + log exp (-S*Z) per ogni elemento
        S0 = numpy.dot(w.T, DTR0) + b
        cxe0 = numpy.logaddexp(0, -(S0*-1)).mean() #calcola log exp(0)=1 + log exp (-S*Z) per ogni elemento
        return pi1*cxe1+(1-pi1)*cxe0 + 0.5*l*numpy.linalg.norm(w)**2
    return logreg_obj #non chiamo la funzione ma la torno soltanto, la chiamerà al suo interno fmin_l_bfgs_b

def PriWeiLinearLogisticRegression(DTR,LTR, s, lamb, pi1):
        logreg_obj = pri_wei_logreg_obj_wrap(DTR, LTR, lamb, pi1)
        _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(2), approx_grad=True)
        _w = _v[0]
        _b = _v[-1]
        S = numpy.dot(_w.T, s) + _b - numpy.log(pi1/(1-pi1))
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

def BalancedLinearLogisticRegression(DTR,LTR, DTE, lamb,pi1):
        logreg_obj = balanced_logreg_obj_wrap(DTR, LTR, lamb,pi1)
        _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0]+1), approx_grad=True)
        _w = _v[0:DTR.shape[0]]
        _b = _v[-1]
        S = numpy.dot(_w.T, DTE) + _b

        return S