import numpy
import matplotlib.pyplot as plt
from Bayes_Decision_Model_Evaluation import BayesDecision
from numpy.lib.function_base import corrcoef

def plot_hist(D, L, hFea, save_name=""):
    """ Plots histograms given D and L which are training/test data and labels and hFea which are the attributes of the dataset,
        store them in the folder called Generated_figures"""
    
    D0 = D[:, L==0] 
    D1 = D[:, L==1] 

    for dIdx in range(11):
        plt.figure()
        plt.xlabel(hFea[dIdx]) 
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'BAD WINES')
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'GOOD WINES')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig('Graphics/Generated_figures/Histograms/hist_%d%s.jpg' % (dIdx, save_name), format='jpg')
    plt.show()

def plot_scatter(D, L, hFea):
    
    """ Plots scatters given D and L which are training/test data and labels and hFea which are the attributes of the dataset,
        store them in the folder called Generated_figures"""
    
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
            plt.savefig('Graphics/Generated_figures/Scatters/scatter_%d_%d.jpg' % (dIdx1, dIdx2), format='jpg')
        plt.show()

def plot_heatmap(D, save_name):
    
    """ Plots correlations given D which are training/test data, store them in the folder called Generated_figures"""
    
    pearson_matrix = corrcoef(D)
    plt.imshow(pearson_matrix, cmap='Purples')
    plt.savefig('Graphics/Generated_figures/Correlations/%s.jpg' % (save_name))
    plt.show()
    return pearson_matrix

def plotDCF(x, y,xlabel):
    """ Plots the minDCF trend when the different applications change, x is the list of lambda, y is the list of minDCF,
        store them in the folder called Generated_figures"""
    
    
    plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5', color='b')
    plt.plot(x, y[len(x): 2*len(x)], label='min DCF prior=0.9', color='r')
    plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=0.1', color='g')
    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    plt.legend(["min DCF prior=0.5", "min DCF prior=0.9", "min DCF prior=0.1"])
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    plt.savefig('Graphics/Generated_figures/DCFPlots/minDCF_%s.jpg' % (xlabel))
    plt.show()
    return

def bayes_error_plot(pArray,llrs,Labels, minCost=False):
    
    """ Plots the bayes error, p in the bound, llr and labels are the log likelihood ratio and the class labels respectively,
        store them in the folder called Generated_figures"""
    
    y=[]
    for p in pArray:
        pi = 1.0/(1.0+numpy.exp(-p))
        if minCost:
            y.append(BayesDecision.compute_min_DCF(llrs, Labels, pi, 1, 1))
        else: 
            y.append(BayesDecision.compute_act_DCF(llrs, Labels, pi, 1, 1))

    return numpy.array(y)
