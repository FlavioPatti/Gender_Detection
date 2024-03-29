import numpy
import matplotlib.pyplot as plt
import scipy.optimize
import utilities as ut


def train_SVM_linear(DTR, LTR, DTE, C, K, balanced = False, pi1=0.5 ):
    """Implementation of the Linear SVM """
    
    DTREXT = numpy.vstack([DTR, K* numpy.ones((1,DTR.shape[1] ))])
    
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    
    H = numpy.dot(DTREXT.T, DTREXT)
    H = ut.vcol(Z)* ut.vrow(Z) * H
    
    def JDual(alpha):
        Ha = numpy.dot(H,ut.vcol(alpha))
        aHa = numpy.dot(ut.vrow(alpha), Ha)
        a1 = alpha.sum()
        return -0.5* aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size) 
                                       
    def LDual(alpha): 
        loss, grad = JDual(alpha)
        return -loss, -grad 
    
    def JPrimal(w): 
        S = numpy.dot(ut.vrow(w), DTREXT) 
        loss = numpy.maximum(numpy.zeros(S.shape), 1-Z*S).sum()
        return 0.5 *numpy.linalg.norm(w)*2 + C * loss
    
    bounds = []
    if balanced == False:
        for i in range(DTR.shape[1]):
            bounds.append((0, C))
    elif balanced == True:
        N = LTR.size #tot number of samples
        n_T = (1*(LTR==1)).sum() #num of samples belonging to the true class
        n_F = (1*(LTR==0)).sum() #num of samples belonging to the false class
        pi_emp_T = n_T / N
        pi_emp_F = n_F / N
        C_T = C * pi1 / pi_emp_T
        C_F = C * (1-pi1) / pi_emp_F 
        for i in range(DTR.shape[1]):
            if LTR[i] == 1:
                bounds.append((0,C_T))
            else:
                bounds.append((0,C_F))
 
    alphaStar, _x, _y, = scipy.optimize.fmin_l_bfgs_b( 
        LDual, numpy.zeros(DTR.shape[1]), bounds = bounds, factr = 0.0, maxiter = 100000, maxfun = 100000
    )
    
    wStar = numpy.dot(DTREXT, ut.vcol(alphaStar)*ut.vcol(Z))
    
    DTEEXT = numpy.vstack([DTE, K* numpy.ones((1,DTE.shape[1] ))])    
     
    S = numpy.dot(wStar.T, DTEEXT)
    S = numpy.hstack(ut.vrow(S))
    
    return S