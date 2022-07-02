import numpy
import matplotlib.pyplot as plt

from Graphics import graphics
from Preprocessing import pca
from Preprocessing import lda
from Preprocessing import pca_lda
from Generative_Models import MultivariateGaussianClassifier
from Generative_Models import NaiveBayesClassifier
from Generative_Models import TiedGaussianClassifier
from Generative_Models import TiedNaiveBayes
from Logistic_Regression import LinearLogisticRegression
from Logistic_Regression import QuadraticLogisticRegression
from Bayes_Decision_Model_Evaluation import BayesDecision
from Preprocessing import Gaussianization


FLAG_TRAINING=1
FLAG_TESTING=0
FLAG_SINGLEFOLD=1
FLAG_KFOLD=0
FLAG_SHOW_FIGURES = 0
FLAG_PCA = 0
FLAG_LDA = 0
FLAG_MVG = 1
FLAG_NAIVE = 0
FLAG_TIED = 0
FLAG_LOGISTIC = 0
FLAG_BAYES_DECISION = 0

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

def split_db_2to1(D, L, seed=0):
    """ Split the dataset in two parts, one is 2/3, the other is 1/3
        first part will be used for model training, second part for validation
        D is the dataset, L the corresponding labels
        returns:
        DTR_T = Dataset for training set
        LTR_T = Labels for training set
        DTR_V = Dataset for validation set
        LTR_V = Labels for validation set
    """
    nTrain = int(D.shape[1]*2.0 / 3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxVal = idx[nTrain:]
    DTR_T = D[:, idxTrain]
    DTR_V = D[:, idxVal]
    LTR_T = L[idxTrain]
    LTR_V = L[idxVal]
    return (DTR_T, LTR_T), (DTR_V, LTR_V)

if __name__ == '__main__':

    #carico i dati
    DTR, LTR, DTE, LTE = load_train_and_test();
    print("Total sample of training set: ", DTR.shape[1] )
    print("Total sample of test set: ", DTE.shape[1] )
    
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
    # 3 applications: main balanced one and two unbalanced
    applications = [[0.5, 1, 1], [0.1, 1, 1], [0.9, 1, 1]]

    if FLAG_TRAINING:
        if FLAG_SINGLEFOLD:
            (DTR_T, LTR_T), (DTR_V, LTR_V)=split_db_2to1(DTR,LTR)
            sample_class0 = (LTR_T==0).sum()
            print("Sample of class 0: ", sample_class0)
            sample_class1 = (LTR_T==1).sum()
            print("Sample of class 1: ", sample_class1, "\n")
            DTR_T0=DTR_T[:,LTR_T==0]
            DTR_T1=DTR_T[:,LTR_T==1]
            print(DTR_T0.shape)
            print(DTR_T1.shape)
            if FLAG_SHOW_FIGURES:
                graphics.plot_hist(DTR_T, LTR_T, hFea)
                graphics.plot_scatter(DTR_T, LTR_T, hFea)
                DTR_T_gauss = Gaussianization.compute_ranking(DTR_T);
                print("dimensione DTR_T gauss")
                print(DTR_T_gauss.shape)
                #DTR_V_gauss= Gaussianization.compute_ranking(DTR_V);
                #print("dimensione DTR_V gauss")
                #print(DTR_V_gauss.shape)
                graphics.plot_hist(DTR_T_gauss, LTR_T, hFea, "gauss");
                #graphics.plot_hist(DTR_V_gauss, LTR_V, hFea, "DTR_V_gauss");
                graphics.plot_heatmap(DTR_T0, "bad_wines_correlation");
                graphics.plot_heatmap(DTR_T1, "good_wines_correlation");
                graphics.plot_heatmap(DTR_T, "global_correlation");


            if FLAG_PCA:
                DTR_T=pca.PCA(DTR_T, LTR_T, 9)
                DTR_V=pca.PCA(DTR_V, LTR_V, 9)
                print("PCA dimensionality: ",DTR_T.shape)

            if FLAG_LDA:
                DTR_T=lda.LDA(DTR_T, LTR_T, 3)
                DTR_V=lda.LDA(DTR_V, LTR_V, 3)
                print("LDA dimensionality: ",DTR_T.shape)

            if FLAG_PCA & FLAG_LDA:
                DTR_T=pca_lda.PCA_LDA(DTR_T, LTR_T, 9)
                DTR_V=pca_lda.PCA_LDA(DTR_V, LTR_V, 9)
                print("PCA-LDA dimensionality: ",DTR_T.shape)

            if FLAG_MVG:
                MultivariateGaussianClassifier.MultivariateGaussianClassifier(DTR_T, LTR_T, DTR_V, LTR_V)

            if FLAG_NAIVE:
                NaiveBayesClassifier.NaiveBayesClassifier(DTR_T, LTR_T, DTR_V, LTR_V)

            if FLAG_TIED:
                TiedGaussianClassifier.TiedGaussianClassifier(DTR_T, LTR_T, DTR_V, LTR_V)
                TiedNaiveBayes.TiedNaiveBayes(DTR_T, LTR_T, DTR_V, LTR_V)

            if FLAG_LOGISTIC:
                LinearLogisticRegression.LinearLogisticRegression(DTR_T, LTR_T, DTR_V, LTR_V)
                QuadraticLogisticRegression.QuadraticLogisticRegression(DTR_T, LTR_T, DTR_V, LTR_V)
                
            if FLAG_BAYES_DECISION:
                BayesDecision.BayesDecision(DTR_T, LTR_T, DTR_V, LTR_V)
                
    if FLAG_TESTING:
        if FLAG_SHOW_FIGURES:
            graphics.plot_hist(DTR, LTR, hFea)
            graphics.plot_scatter(DTR, LTR, hFea)
            DTR_gauss = Gaussianization.compute_ranking(DTR);
            print("dimensione DTR gauss")
            print(DTR_gauss.shape)
            #DTE_gauss= Gaussianization.compute_ranking(DTE);
            #print("dimensione DTE gauss")
            #print(DTE_gauss.shape)
            graphics.plot_hist(DTR_gauss, LTR, hFea, "gauss");
            #graphics.plot_hist(DTE_gauss, LTE, hFea, "DTE_gauss");
            graphics.plot_heatmap(DTR0, "bad_wines_correlation");
            graphics.plot_heatmap(DTR1, "good_wines_correlation");
            graphics.plot_heatmap(DTR, "global_correlation");

        
        if FLAG_PCA:
            DTR=pca.PCA(DTR, LTR, 9)
            DTE=pca.PCA(DTE, LTE, 9)
            print("PCA dimensionality: ",DTR.shape)

        if FLAG_LDA:
            DTR=lda.LDA(DTR, LTR, 3)
            DTE=lda.LDA(DTE, LTE, 3)
            print("LDA dimensionality: ",DTR.shape)

        if FLAG_PCA & FLAG_LDA:
            DTR=pca_lda.PCA_LDA(DTR, LTR, 9)
            DTE=pca_lda.PCA_LDA(DTE, LTE, 9)
            print("PCA-LDA dimensionality: ",DTR.shape)
        
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
    
    
    
    