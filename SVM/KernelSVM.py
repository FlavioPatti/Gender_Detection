import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

def mcol(v):
    return v.reshape((v.size,1))

def mrow(v):
    return v.reshape((1,v.size))

# Compute the kernel dot-product
def kernel(x1, x2, type, d = 0, c = 0, gamma = 0, csi = 1): #csi = 1 --> eps = ksi^2...... c = [0,1]... gamma = [1.0, 2.0]
    """Implementation of kernels"""
    
    if type == "poly":
        # Polynomial kernel of degree d
        return (np.dot(x1.T, x2) + c) ** d + csi**2
    
    elif type == "RBF":
        # Radial Basic Function kernel
        dist = mcol((x1**2).sum(0)) + mrow((x2**2).sum(0)) - 2 * np.dot(x1.T, x2)
        k = np.exp(-gamma * dist) + csi**2
        return k
    
 
def quad_kernel_svm(DTR, LTR, DTE, C, c=0,gamma=0,csi=0, type="poly", balanced = False):
    """Implementation of the quadratic svm"""
 
    x0 = np.zeros(DTR.shape[1])
    d = 2
    
    
    N = LTR.size #tot number of samples
    n_T = (1*(LTR==1)).sum() #num of samples belonging to the true class
    n_F = (1*(LTR==0)).sum() #num of samples belonging to the false class
    pi_emp_T = n_T / N
    pi_emp_F = n_F / N
    
    C_T = C * 0.5 / pi_emp_T
    C_F = C * (1-0.5) / pi_emp_F 
 
    bounds = [(0,1)] * LTR.size
 
    if balanced == True:
        
        for i in range (LTR.size):
            if (LTR[i]==1):
                bounds[i] = (0,C_T)
            else :
                bounds[i] = (0,C_F)
                
    if balanced == False:
        
        for i in range (LTR.size):
            bounds[i]=(0,C)
    
    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
 
    H = None
 
    if type == "poly":
        H = mcol(Z) * mrow(Z) * ((np.dot(DTR.T, DTR) + c) ** d + csi**2)  #type == poly
    elif type == "RBF":
        dist = mcol((DTR**2).sum(0)) + mrow((DTR**2).sum(0)) - 2 * np.dot(DTR.T, DTR)
        H = np.exp(-gamma * dist) + csi**2
        H = mcol(Z) * mrow(Z) * H
 
    def JDual(alpha):
        Ha = np.dot(H, alpha.T)
        aHa = np.dot(alpha, Ha)
        a1 = alpha.sum()
        return 0.5 * aHa - a1, Ha - np.ones(alpha.size)
    
    def LDual(alpha):
        loss, grad = JDual(alpha)
        return loss, grad
    
    x,_,_ = scipy.optimize.fmin_l_bfgs_b(LDual, x0, factr=0.0, approx_grad=False, bounds=bounds, maxfun=100000, maxiter=100000)
 
    #we are not able to compute the primal solution, but we can still compute the scores like that
    S = np.sum((x*Z).reshape([DTR.shape[1],1]) * kernel(DTR, DTE, type, d, c, gamma, csi), axis=0)
    return S.reshape(S.size,)