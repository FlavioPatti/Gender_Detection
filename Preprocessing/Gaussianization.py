import numpy
import matplotlib.pyplot as plt
import scipy


def compute_ranking(D):
    N = D.shape[1]
    ranks = []
    for j in range(D.shape[0]):
        tempSum = 0
        for i in range(D.shape[1]):
            tempSum += (D[j, :] < D[j,i]).astype(int)
        tempSum += 1;
        ranks.append(tempSum/(N+2))
    D_gauss = scipy.stats.norm.ppf(ranks)

        
    return D_gauss

