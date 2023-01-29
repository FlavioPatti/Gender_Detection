import numpy
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

def LinearLogisticRegression(DTR,LTR, DTE, lamb, balanced=False, pi1=0.5):
    """Implementation of the Linear Logistic Regression"""
    if balanced==False:
        logreg_obj = logreg_obj_wrap(DTR, LTR, lamb)
    else: 
        logreg_obj = balanced_logreg_obj_wrap(DTR, LTR, lamb,pi1)
    
    _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0]+1), approx_grad=True)
    _w = _v[0:DTR.shape[0]]
    _b = _v[-1]
    S = numpy.dot(_w.T, DTE) + _b

    return S




"""
Implementation of the calibration/fusion using the prior weighted logistic regression.
Forced to use another functions instead of the balanced one because in this case we work with VECTORS of scores,
not MATRICES of samples
"""
def PriWeiLinearLogisticRegression(STR,LTR, STE, lamb, pi1):
    
    logreg_obj = pri_wei_logreg_obj_wrap(STR, LTR, lamb, pi1)
    _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(2), approx_grad=True)
    _w = _v[0]
    _b = _v[-1]
    Sfin = numpy.dot(_w.T, STE) + _b - numpy.log(pi1/(1-pi1))
    return Sfin

def pri_wei_logreg_obj_wrap(STR, LTR, l, pi1):
    STR1=STR[LTR==1]
    STR0=STR[LTR==0]
    def logreg_obj(v): #v packs w b
        w = v[0]
        b = v[-1]
        S1 = numpy.dot(w.T, STR1) + b
        cxe1 = numpy.logaddexp(0, -S1).mean() 
        S0 = numpy.dot(w.T, STR0) + b
        cxe0 = numpy.logaddexp(0, -(S0*-1)).mean() 
        return pi1*cxe1+(1-pi1)*cxe0 + 0.5*l*numpy.linalg.norm(w)**2
    return logreg_obj 
