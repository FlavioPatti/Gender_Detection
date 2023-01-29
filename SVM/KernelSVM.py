import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import utilities as ut

# Compute the kernel dot-product
def kernel(x1, x2, type, d = 0, c = 0, gamma = 0, csi = 1):
    """Implementation of 2 types of kernels:
    -Polynomial kernel: k(x1, x2) = (x1^T x2 + c)^d
    -Radial Basic Funcion kernel: k(x1, x2) = e^(-gamma*∥x1- x2∥)^2
    """
    
    if type == "poly":
        # Polynomial kernel of degree d
        return (np.dot(x1.T, x2) + c) ** d + csi**2
    
    elif type == "RBF":
        # Radial Basic Function kernel
        dist = ut.vcol((x1**2).sum(0)) + ut.vrow((x2**2).sum(0)) - 2 * np.dot(x1.T, x2)
        k = np.exp(-gamma * dist) + csi**2
        return k
    
 
def kernel_svm(DTR, LTR, DTE, C, c=0,gamma=0,csi=0, type="poly", balance_data = False, pi=0.5):
    """Implementation of the kernel svm"""
    
    tot_samples = LTR.size #num of total samples
    num_samples_true = (1*(LTR==1)).sum() #num of total samples from class true
    num_samples_false = (1*(LTR==0)).sum() #num of total samples from class false
    pi_emp_true = num_samples_true / tot_samples
    pi_emp_false = num_samples_false / tot_samples
    
    C_true = C * pi / pi_emp_true
    C_false = C * (1-pi) / pi_emp_false 
 
    bounds = [(0,1)] * LTR.size
 
    if balance_data == True:
        
        for i in range (LTR.size):
            if (LTR[i]==1):
                bounds[i] = (0,C_true)
            else :
                bounds[i] = (0,C_false)
                
    else:
        
        for i in range (LTR.size):
            bounds[i]=(0,C)
    
    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
 
    H = None
    d = 2
    if type == "poly":
        H = ut.vcol(Z) * ut.vrow(Z) * kernel(DTR, DTR, type, d, c, gamma, csi) #type == poly
    elif type == "RBF":
        H = ut.vcol(Z) * ut.vrow(Z) * kernel(DTR, DTR, type, d, c, gamma, csi) #tipe == RBF
 
    def JDual(alpha):
        Ha = np.dot(H, alpha.T)
        aHa = np.dot(alpha, Ha)
        a1 = alpha.sum()
        return 0.5 * aHa - a1, Ha - np.ones(alpha.size)
    
    def LDual(alpha):
        loss, grad = JDual(alpha)
        return loss, grad
    
    x,_,_ = scipy.optimize.fmin_l_bfgs_b(LDual, np.zeros(DTR.shape[1]), factr=0.0, approx_grad=False, bounds=bounds, maxfun=100000, maxiter=100000)
 
    #we are not able to compute the primal solution, but we can still compute the scores like that
    S = np.sum((x*Z).reshape([DTR.shape[1],1]) * kernel(DTR, DTE, type, d, c, gamma, csi), axis=0)
    return S.reshape(S.size,)