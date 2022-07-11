import matplotlib.pyplot as plt
import scipy

"""Compute the Gaussianization of the dataset (mapping a set of features to values whose empirical cumulative distribution function
is well approximated by a Gaussian c.d.f.)"""

def Gaussianization(DTR,DTE):
    N = DTR.shape[1]
    ranks = []
    for j in range(DTR.shape[0]):
        tempSum = 0
        for i in range(DTR.shape[1]):
            tempSum += (DTE[j, :] < DTR[j,i]).astype(int)
        tempSum += 1;
        ranks.append(tempSum/(N+2))
    D_gauss = scipy.stats.norm.ppf(ranks)

        
    return D_gauss

