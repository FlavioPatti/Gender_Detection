import numpy
import matplotlib.pyplot as plt
import scipy.optimize

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
        cxe = numpy.logaddexp(0, -S*Z).mean() 
        return cxe + 0.5*l*numpy.linalg.norm(w)**2
    return logreg_obj 

def LinearLogisticRegression(DTR,LTR, DTE, lamb):
    """Implementation of the Linear Logistic Regression"""
    
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
        cxe1 = numpy.logaddexp(0, -S1).mean() 
        S0 = numpy.dot(w.T, DTR0) + b
        cxe0 = numpy.logaddexp(0, -(S0*-1)).mean() 
        return pi1*cxe1+(1-pi1)*cxe0 + 0.5*l*numpy.linalg.norm(w)**2
    return logreg_obj 

def PriWeiLinearLogisticRegression(s,sl, S, lamb, pi1):
    """Implementation of the calibration using the prior weighted logistic regression"""
    
    logreg_obj = pri_wei_logreg_obj_wrap(s, sl, lamb, pi1)
    _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(2), approx_grad=True)
    _w = _v[0]
    _b = _v[-1]
    Sfin = numpy.dot(_w.T, S) + _b - numpy.log(pi1/(1-pi1))
    return Sfin

def FusionLinearLogisticRegression(s,sl, S,s1,sl1,S1, lamb, pi1):
    """Implementation of the fusion of the scores of two different classifiers"""
    
    logreg_obj = pri_wei_logreg_obj_wrap(s, sl, lamb, pi1)
    _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(2), approx_grad=True)
    _w = _v[0]
    _b = _v[-1]
    
    logreg_obj1 = pri_wei_logreg_obj_wrap(s1, sl1, lamb, pi1)
    _v1, _J1, _d1 = scipy.optimize.fmin_l_bfgs_b(logreg_obj1, numpy.zeros(2), approx_grad=True)
    _w1 = _v1[0]
    _b1 = _v1[-1]
    Sfin = numpy.dot(_w.T, S) + numpy.dot(_w1.T, S1) + _b +_b1
    return Sfin

def balanced_logreg_obj_wrap(DTR, LTR, l,pi1):
    DTR1=DTR[:,LTR==1]
    DTR0=DTR[:,LTR==0]
    M = DTR.shape[0]
    def logreg_obj(v): #v packs w b
        w = mcol(v[0:M])
        b = v[-1]
        S1 = numpy.dot(w.T, DTR1) + b
        cxe1 = numpy.logaddexp(0, -S1).mean() 
        S0 = numpy.dot(w.T, DTR0) + b
        cxe0 = numpy.logaddexp(0, -(S0*-1)).mean() 
        return pi1*cxe1+(1-pi1)*cxe0 + 0.5*l*numpy.linalg.norm(w)**2
    return logreg_obj 

def BalancedLinearLogisticRegression(DTR,LTR, DTE, lamb,pi1):
    """Implementation of the balanced linear logistic regression"""
    
    logreg_obj = balanced_logreg_obj_wrap(DTR, LTR, lamb,pi1)
    _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0]+1), approx_grad=True)
    _w = _v[0:DTR.shape[0]]
    _b = _v[-1]
    S = numpy.dot(_w.T, DTE) + _b

    return S