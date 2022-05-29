import numpy
import matplotlib.pyplot as plt

from Graphics import graphics
from PCA import pca
from LDA import lda
from PCA_LDA import pca_lda
from Generative_Models import MultivariateGaussianClassifier
from Generative_Models import NaiveBayesClassifier
from Generative_Models import TiedGaussianClassifier
from Generative_Models import TiedNaiveBayes
from Logistic_Regression import LinearLogisticRegression
from Logistic_Regression import QuadraticLogisticRegression
from Bayes_Decision_Model_Evaluation import BayesDecision


FLAG_SHOW_FIGURES = 0
FLAG_PCA = 0
FLAG_LDA = 0
FLAG_MVG = 0
FLAG_NAIVE = 0
FLAG_TIED = 0
FLAG_LOGISTIC = 0
FLAG_BAYES_DECISION = 1

def vcol(v):
    return v.reshape((v.size,1))

def vrow(v):
    return v.reshape((1,v.size))

def load_train_and_test():
    
    DTR = []
    LTR = []

    f=open('data/Train.txt', encoding="ISO-8859-1")

    for line in f:
        line = line.strip().split(',')
        sample = vcol(numpy.array(line[0:11], dtype=numpy.float32))
        DTR.append(sample)
        LTR.append(line[11])
    f.close()
    
    DTE = []
    LTE = []
    
    f=open('data/Test.txt', encoding="ISO-8859-1")
    
    for line in f:
        line = line.strip().split(',')
        sample = vcol(numpy.array(line[0:11], dtype=numpy.float32))
        DTE.append(sample)
        LTE.append(line[11])
    f.close()
    
    return numpy.hstack(DTR), numpy.array(LTR, dtype=numpy.int32), numpy.hstack(DTE), numpy.array(LTE, dtype=numpy.int32)  


if __name__ == '__main__':

    #carico i dati
    DTR, LTR, DTE, LTE = load_train_and_test();
    #print(DTR, LTR)
    print("Campioni totali del dataset: ", DTR.shape[1] )
    sample_class0 = (LTR==0).sum()
    print("Campioni classe 0: ", sample_class0)
    sample_class1 = (LTR==1).sum()
    print("Campioni classe 1: ", sample_class1, "\n")
    
    hFea = {
        0: 'Fixed Acidity',
        1: 'Volatile Acidity',
        2: 'Citric Acid',
        3: 'Residual Sugar',
        4:  'Chlorides',
        5: 'Free Sulfur Dioxide',
        6: 'Total Sulfur Dioxide',
        7: 'Density',
        8: 'PH',
        9: 'Sulphates',
        10: 'Alcohol'
        }
    
    if FLAG_SHOW_FIGURES:
        graphics.plot_hist(DTR, LTR, hFea)
        graphics.plot_scatter(DTR, LTR, hFea)

    
    if FLAG_PCA:
        pca.PCA(DTR, LTR, 2)
    
    if FLAG_LDA:
        lda.LDA(DTR, LTR, 2)
    
    if FLAG_PCA & FLAG_LDA:
        pca_lda.PCA_LDA(DTR, LTR, 2)
    
    if FLAG_MVG:
        MultivariateGaussianClassifier.MultivariateGaussianClassifier(DTR, LTR, DTE, LTE)
    
    if FLAG_NAIVE:
        NaiveBayesClassifier.NaiveBayesClassifier(DTR, LTR, DTE, LTE)
    
    if FLAG_TIED:
        TiedGaussianClassifier.TiedGaussianClassifier(DTR, LTR, DTE, LTE)
        TiedNaiveBayes.TiedNaiveBayes(DTR, LTR, DTE, LTE)
    
    if FLAG_LOGISTIC:
        LinearLogisticRegression.LinearLogisticRegression(DTR, LTR, DTE, LTE)
        QuadraticLogisticRegression.QuadraticLogisticRegression(DTR, LTR, DTE, LTE)
        
    if FLAG_BAYES_DECISION:
        BayesDecision.BayesDecision(DTR, LTR, DTE, LTE)
    
    
    
    