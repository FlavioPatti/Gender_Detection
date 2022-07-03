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


def QuadraticLogisticRegression(DTR, LTR, DTE, LTE, l): 
    
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
    LP = S>0
    
    accuracy = 0
    for i in range(LTE.shape[0]):
        if(LP[i]==LTE[i]):
            accuracy+=1
            
    accuracy = accuracy / DTE_exp.shape[1]
    errore = 1 - accuracy
    
    print("Accuracy for the Quadratic Logistic Regression: ")
    print("%.2f" % (100*accuracy) )
    print("Error for the Quadratic Logistic Regression: ")
    print("%.2f\n" % (100*errore) )

    return S
        
   

