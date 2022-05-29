import matplotlib.pyplot as plt
from PCA import pca
from LDA import lda

def PCA_LDA(D,L, m):
    DP = pca.PCA(D,L, m)
    DW_0, DW_1 = lda.LDA(DP, L, m)
    
    plt.figure()
    plt.title('PCA + LDA')
    plt.scatter(DW_0[0], DW_0[1], label = "BAD WINES")
    plt.scatter(DW_1[0], DW_1[1], label = "GOOD WINES")
    plt.legend()
    plt.show()
    
    