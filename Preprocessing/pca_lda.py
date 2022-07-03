import matplotlib.pyplot as plt
from Preprocessing import pca
from Preprocessing import lda

def PCA_LDA(D,L, m):
    DP = pca.PCA(D,L, m)
    DW = lda.LDA(DP, L, 1)
    return DW
    
    
    