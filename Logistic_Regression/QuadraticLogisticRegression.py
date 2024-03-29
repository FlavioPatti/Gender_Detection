import numpy
import matplotlib.pyplot as plt
import scipy.optimize
import utilities as ut

def logreg_obj_wrap(DTR, LTR, l):
    Z = LTR * 2.0 - 1.0
    M = DTR.shape[0]
    def logreg_obj(v): #v packs w b
        w = ut.vcol(v[0:M])
        b = v[-1]
        S = numpy.dot(w.T, DTR) + b
        cxe = numpy.logaddexp(0, -S*Z).mean() 
        return cxe + 0.5*l*numpy.linalg.norm(w)**2
    return logreg_obj

def balanced_logreg_obj_wrap(DTR, LTR, l,pi1):
    DTR1=DTR[:,LTR==1]
    DTR0=DTR[:,LTR==0]
    M = DTR.shape[0]
    def logreg_obj(v): #v packs w b
        w = ut.vcol(v[0:M])
        b = v[-1]
        S1 = numpy.dot(w.T, DTR1) + b
        cxe1 = numpy.logaddexp(0, -S1).mean()
        S0 = numpy.dot(w.T, DTR0) + b
        cxe0 = numpy.logaddexp(0, -(S0*-1)).mean() 
        return pi1*cxe1+(1-pi1)*cxe0 + 0.5*l*numpy.linalg.norm(w)**2
    return logreg_obj 
     


def QuadraticLogisticRegression(DTR, LTR, DTE, lamb, balanced=False, pi1=0.5): 
    """Implementation of the quadratic logistic regression"""
   
    DTR_exp = [] 
    
    for i in range(DTR.shape[1]):
        x = DTR[:,i:i+1]
        tmp = numpy.dot(x, x.T)
        tmp2 = numpy.ravel(tmp, order='F')
        tmp3 = numpy.concatenate( (ut.vcol(tmp2), ut.vcol(x)), axis=0)
        DTR_exp.append(tmp3)
    DTR_exp=numpy.hstack(DTR_exp)
    
    DTE_exp = []
    
    for i in range(DTE.shape[1]):
        x = DTE[:,i:i+1]
        tmp = numpy.dot(x, x.T)
        tmp2 = numpy.ravel(tmp, order='F')
        tmp3 = numpy.concatenate( (ut.vcol(tmp2), ut.vcol(x)), axis=0)
        DTE_exp.append(tmp3)
    DTE_exp=numpy.hstack(DTE_exp)

    if balanced==False:
        logreg_obj = logreg_obj_wrap(DTR_exp, LTR, lamb)
    else: 
        logreg_obj = balanced_logreg_obj_wrap(DTR_exp, LTR, lamb,pi1)
    
    _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR_exp.shape[0]+1), approx_grad=True)
    _w = _v[0:DTR_exp.shape[0]]
    _b = _v[-1]
    S = numpy.dot(_w.T, DTE_exp) + _b 
    return S



