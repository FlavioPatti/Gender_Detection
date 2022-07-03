import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import sklearn.datasets

def mcol(v):
    return v.reshape((v.size,1))

def mrow(v):
    return v.reshape((1,v.size))

def polynomial_kernel(X1, X2, c, d, gamma):
    """
        Implementation of polynomial kernel function
        k(X1, X2) = (X1.T @ X2 + C) ^ d
        the parameter gamma is not used, it is for compatibility
        with other kernel functions
    """
    return (np.dot(X1.T, X2) + c) ** d


def RBF_kernel(X1, X2, c, d, gamma):
    """
        Implementeation of RBF kernel function
        k(X1, X2) = e ^ (-gamma (norm(X1 - X2)) ^ 2)
        the parameters c and d are not used, it is for compatibility
        with other kernel functions
    """
    X1 = X1.T
    X2 = X2.T
    # return np.exp(-gamma*np.sum((X2-X1[:,np.newaxis])**2, axis=-1))

    # optimized version using property of norm:
    # ||X1-X2||^2 = ||X1||^2 + ||X2||^2 - 2 * X1.T * Y
    X1_norm = np.sum(X1 ** 2, axis=-1)
    X2_norm = np.sum(X2 ** 2, axis=-1)
    return np.exp(-gamma * (X1_norm[:, None] + X2_norm[None, :] - 2 * np.dot(X1, X2.T)))


def svm_dual_kernel_wrapper(DTR, LTR, kernel, K, c, d, gamma):
    """
        Wrapper for the svm_dual_kernel 
        to pass to it the parameters DTR, LTR and K
    """
    def svm_dual_kernel(alpha):
        """ 
            Computes the dual SVM solution using kernel function
            returns L_D_hat = - J_D_hat, and the gradient of L_D_hat
        """
        N = DTR.shape[1]

        z = mcol(np.array(2 * LTR - 1))

        # + K^2 (=Xi) for regularized bias in non-linear svm version
        H_hat = z * z.T * (kernel(DTR, DTR, c, d, gamma) + K**2)

        # J_D_hat = -1/2 * np.dot(np.dot(alpha.T, H_hat), alpha) + \
        #     np.dot(alpha.T, np.ones(N))

        # need to use multi_dot because it gives numerical problems otherwise
        J_D_hat = -1/2 * np.linalg.multi_dot([alpha.T, H_hat, alpha]) + \
            np.dot(alpha.T, np.ones(N))

        L_D_hat = - J_D_hat
        grad_L_D_hat = mcol(np.dot(H_hat, alpha) - np.ones(N))

        return L_D_hat, grad_L_D_hat
    return svm_dual_kernel



def svm_kernel_polynomial(DTR, LTR, DTE, K, C, c, d=2):
    """ Implementation of the svm classifier using polynomial kernel function """
    N = DTR.shape[1]
    # starting point
    x0 = np.zeros(N)
    bounds = []
    for i in range(N):
        bounds.append((0, C))

    svm_dual_kernel = svm_dual_kernel_wrapper(
        DTR, LTR, polynomial_kernel, K, c, d, 0)

    x, f, d_ = scipy.optimize.fmin_l_bfgs_b(svm_dual_kernel, x0, factr=1.0, bounds=bounds)

    # Compute scores of test samples
    S = np.empty(DTE.shape[1])
    z = mcol(np.array(2 * LTR - 1))

    for t in range(DTE.shape[1]):
        for i in range(N):
            if(mcol(x)[i] > 0):
                S[t] += mcol(x)[i] * z[i] * \
                    (polynomial_kernel(DTR.T[i], DTE.T[t], c, d, 0) + K**2)
    # + K^2 (=Xi) for regularized bias in non-linear svm version
    
    return S.reshape(S.size,)



def svm_kernel_RBF(DTR, LTR, DTE, K, C, g):
    """ Implementation of the svm classifier using RBF kernel function """
    N = DTR.shape[1]
    # starting point
    x0 = np.zeros(N)
    bounds = []
    for i in range(N):
        bounds.append((0, C))

    svm_dual_kernel = svm_dual_kernel_wrapper(DTR, LTR, RBF_kernel, K, 0, 0, g)

    x, f, d_ = scipy.optimize.fmin_l_bfgs_b(svm_dual_kernel, x0, factr=1.0, bounds=bounds)

    # Compute scores of test samples
    S = np.empty(DTE.shape[1])
    z = mcol(np.array(2 * LTR - 1))

    for t in range(DTE.shape[1]):
        for i in range(N):
            if(mcol(x)[i] > 0):
                S[t] += mcol(x)[i] * z[i] * (RBF_kernel(mcol(DTR.T[i]),
                                                        mcol(DTE.T[t]), 0, 0, g) + K**2)
    # + K^2 (=Xi) for regularized bias in non-linear svm version
    return S.reshape(S.size,)
