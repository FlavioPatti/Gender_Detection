import matplotlib.pyplot as plt
from scipy.stats import norm

"""Compute the Gaussianization of the dataset (mapping a set of features to values whose empirical cumulative distribution function
is well approximated by a Gaussian c.d.f.)"""

def Gaussianization(TD, D):
    if (TD.shape[0]!=D.shape[0]):
        print("Datasets not aligned in dimensions")
    ranks=[]
    for j in range(D.shape[0]):
        tempSum=0
        for i in range(TD.shape[1]):
            tempSum+=(D[j, :]<TD[j, i]).astype(int)
        tempSum+=1
        ranks.append(tempSum/(TD.shape[1]+2))
    y = norm.ppf(ranks)
    return y
