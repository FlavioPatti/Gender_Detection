import numpy
import matplotlib.pyplot as plt

def plot_hist(D, L, hFea):
    
    D0 = D[:, L==0] 
    D1 = D[:, L==1] 

    for dIdx in range(11):
        plt.figure()
        plt.xlabel(hFea[dIdx]) #scrive sull'asse delle x delle figure i vari attributi di hFea
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'BAD WINES')
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'GOOD WINES')
        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.savefig('Graphics/Generated_figures/hist_%d.jpg' % dIdx, format='jpg')
    plt.show()

def plot_scatter(D, L, hFea):
    #stampa tutti gli altri pdf mettendo sugli assi x e y i vari attributi
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    for dIdx1 in range(11): 
        for dIdx2 in range(11):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'BAD WINES')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'GOOD WINES')
        
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            plt.savefig('Graphics/Generated_figures/scatter_%d_%d.jpg' % (dIdx1, dIdx2), format='jpg')
        plt.show()