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
from SVM import LinearSVM
from SVM import KernelSVM
from GMM import gmm
from Bayes_Decision_Model_Evaluation import BayesDecision
from Preprocessing import Gaussianization


FLAG_TRAINING=1
FLAG_TESTING=0
FLAG_SINGLEFOLD=0
FLAG_KFOLD=1
FLAG_SHOW_FIGURES = 0
FLAG_PCA = 0
FLAG_LDA = 0
FLAG_MVG = 0
FLAG_NAIVE = 0
FLAG_TIED = 0
FLAG_LOGISTIC = 0
FLAG_SVM= 0
FLAG_GMM=1
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

def k_fold(D, L, K, algorithm, params=None, seed=0):
    """ Implementation of the k-fold cross validation approach
        D is the dataset, L the labels, K the number of folds
        pi1, Cfn, Cfp are the parameters of the application
        algorithm is the algorithm used as classifier
        params are the additional parameters like hyperparameters
        return the llr and labels
    """
    sizePartitions = int(D.shape[1]/K)
    numpy.random.seed(seed)

    # permutate the indexes of the samples
    idx_permutation = numpy.random.permutation(D.shape[1])

    # put the indexes inside different partitions
    idx_partitions = []
    for i in range(0, D.shape[1], sizePartitions):
        idx_partitions.append(list(idx_permutation[i:i+sizePartitions]))

    all_llr = []
    all_labels = []

    # for each fold, consider the ith partition in the test set
    # the other partitions in the train set
    for i in range(K):
        # keep the i-th partition for test
        # keep the other partitions for train
        idx_test = idx_partitions[i]
        idx_train = idx_partitions[0:i] + idx_partitions[i+1:]

        # from lists of lists collapse the elemnts in a single list
        idx_train = sum(idx_train, [])

        # partition the data and labels using the already partitioned indexes
        DTR = D[:, idx_train]
        DTE = D[:, idx_test]
        LTR = L[idx_train]
        LTE = L[idx_test]

        # calculate scores
        if params is not None:
            llr = algorithm(DTR, LTR, DTE, *params)
        else:
            llr = algorithm(DTR, LTR, DTE)

        # add scores and labels for this fold in total
        all_llr.append(llr)
        all_labels.append(LTE)

    all_llr = numpy.hstack(all_llr)
    all_labels = numpy.hstack(all_labels)
    print(all_llr.shape)
    print(all_labels.shape)

    # if algorithm == logistic_regression:
    # We can recover log-likelihood ratios by subtracting from the score s
    # the empirical prior log-odds of the training set (slide 31)
    #all_llr = all_llr - log(pi1/ (1-pi1))

    #DCF_min = minimum_detection_cost(all_llr, all_labels, pi1, Cfn, Cfp)
    
    return all_llr, all_labels


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
    
    DTR0=DTR[:,LTR==0]
    DTR1=DTR[:,LTR==1]
    print(DTR0.shape)
    print(DTR1.shape)
    
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

    if FLAG_TRAINING:
        if FLAG_SINGLEFOLD:
            (DTR_T, LTR_T), (DTR_V, LTR_V)=split_db_2to1(DTR,LTR)
            sample_class0 = (LTR_T==0).sum()
            print("Sample of class 0: ", sample_class0)
            sample_class1 = (LTR_T==1).sum()
            print("Sample of class 1: ", sample_class1, "\n")
            
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

            if FLAG_SVM:
                K_list = [1, 10];
                C_list = [0.1, 1.0, 10.0];
                for K in K_list:
                    for C in C_list:
                        print("SVM Linear: K = %f, C = %f" % (K,C), "\n")
                        LinearSVM.train_SVM_linear(DTR_T,LTR_T, DTR_V, LTR_V, K,C);   
                c_list = [0, 1]
                for K in K_list:
                    for C in C_list:
                        for c in c_list:
                            print("SVM Polynomial Kernel: K = %f, C = %f, d=2, c= %f" % (K,C,c), "\n")
                            KernelSVM.svm_kernel_polynomial(DTR_T,LTR_T, DTR_V, K, C, d=2, c=c);          
                g_list = [1,10]
                for K in K_list:
                    for C in C_list:
                        for g in g_list:
                            print("SVM RBF Kernel: K = %f, C = %f, g=%f" % (K,C,g), "\n")
                            KernelSVM.svm_kernel_RBF(DTR_T,LTR_T, DTR_V, K, C, g= g);
                
            if FLAG_GMM:
                psi = 0.01
                M_list = [2, 4, 8]
                versions = ["full", "diagonal", "tied"]
                for version in versions:
                    for M in M_list:
                        print("GMM version = %s, M = %d, psi = %f" % (version, M, psi), "\n")
                        gmm.GMM_classifier(DTR_T, LTR_T, DTR_V, M, psi, version = version)
                
            if FLAG_BAYES_DECISION:
                BayesDecision.BayesDecision(DTR_T, LTR_T, DTR_V, LTR_V)
            
        K = 5;
        if FLAG_KFOLD: 
            
            if FLAG_MVG:
                print("mvg")
                all_llr, all_labels = k_fold(DTR, LTR, K, MultivariateGaussianClassifier.MultivariateGaussianClassifier)
                print("llr: ", all_llr)
                print("label: ", all_labels)

            if FLAG_NAIVE:
                print("naive")
                all_llr, all_labels = k_fold(DTR, LTR, K, NaiveBayesClassifier.NaiveBayesClassifier)
                print("llr: ", all_llr)
                print("label: ", all_labels)

            if FLAG_TIED:
                print("tied gaussian")
                all_llr, all_labels = k_fold(DTR, LTR, K,  TiedGaussianClassifier.TiedGaussianClassifier)
                print("llr: ", all_llr)
                print("label: ", all_labels)
                
                print("tied naive")
                all_llr, all_labels = k_fold(DTR, LTR, K,  TiedNaiveBayes.TiedNaiveBayes)
                print("llr: ", all_llr)
                print("label: ", all_labels)

            if FLAG_LOGISTIC:
                lambda_list = [0., 1e-6, 1e-3, 1.]
                for l in lambda_list:
                    print(" linear logistic regression with lamb ", l)
                    all_llr, all_labels = k_fold(DTR, LTR, K, LinearLogisticRegression.LinearLogisticRegression, (l,))
                    print("llr: ", all_llr)
                    print("label: ", all_labels)
            
                    print(" quadratic logistic regression with lamb ", l)
                    all_llr, all_labels = k_fold(DTR, LTR, K,  QuadraticLogisticRegression.QuadraticLogisticRegression, (l,))
                    print("llr: ", all_llr)
                    print("label: ", all_labels)

            if FLAG_SVM:
            
                K_list = [1, 10];
                C_list = [0.1, 1.0, 10.0];
                for K_ in K_list:
                    for C in C_list:
                        print("SVM Linear: K = %f, C = %f" % (K_,C), "\n")
                       # all_llr, all_labels = k_fold(DTR, LTR, K, LinearSVM.train_SVM_linear, (K_,C) )   
           
                K_list = [1, 10]
                C_list = [0.1, 1.0, 10.0]       
                c_list = [0, 1]
                for K_ in K_list:
                    for C in C_list:
                        for c in c_list:
                            print("SVM Polynomial Kernel: K = %f, C = %f, d=2, c= %f" % (K_,C,c), "\n")
                            #all_llr, all_labels = k_fold(DTR, LTR, K, KernelSVM.svm_kernel_polynomial, (K_,C, c) );   
                            
                K_list = [1, 10]
                C_list = [0.1, 1.0, 10.0]                 
                g_list = [1,10]
                for K_ in K_list:
                    for C in C_list:
                        for g in g_list:
                            print("SVM RBF Kernel: K = %f, C = %f, g=%f" % (K_,C,g), "\n")
                            all_llr, all_labels = k_fold(DTR, LTR, K, KernelSVM.svm_kernel_RBF, (K_, C, g) ) ;
                          
                
            if FLAG_GMM:
                psi = 0.01
                M_list = [2, 4, 8]
                versions = ["full", "diagonal", "tied"]
                for version in versions:
                    for M in M_list:
                        print("GMM version = %s, M = %d, psi = %f" % (version, M, psi), "\n")
                        all_llr, all_labels = k_fold(DTR, LTR, K, gmm.GMM_classifier, (M, psi, version) )
                
            if FLAG_BAYES_DECISION:
                BayesDecision.BayesDecision(DTR_T, LTR_T, DTR_V, LTR_V)
                 

    if FLAG_TESTING:
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

        if FLAG_SVM:
            K_list = [1, 10];
            C_list = [0.1, 1.0, 10.0];
            for K in K_list:
                for C in C_list:
                    print("SVM Linear: K = %f, C = %f" % (K,C), "\n")
                    LinearSVM.train_SVM_linear(DTR,LTR, DTE, LTE, K,C);   
            c_list = [0, 1]
            for K in K_list:
                for C in C_list:
                    for c in c_list:
                        print("SVM Polynomial Kernel: K = %f, C = %f, d=2, c= %f" % (K,C,c), "\n")
                        KernelSVM.svm_kernel_polynomial(DTR,LTR, DTE, K, C, d=2, c=c);          
            g_list = [1,10]
            for K in K_list:
                for C in C_list:
                    for g in g_list:
                        print("SVM RBF Kernel: K = %f, C = %f, g=%f" % (K,C,g), "\n")
                        KernelSVM.svm_kernel_RBF(DTR,LTR, DTE, K, C, g= g);
            
        if FLAG_GMM:
            psi = 0.01
            M_list = [2, 4, 8]
            versions = ["full", "diagonal", "tied"]
            for version in versions:
                for M in M_list:
                    print("GMM version = %s, M = %d, psi = %f" % (version, M, psi), "\n")
                    gmm.GMM_classifier(DTR, LTR, DTE, M, psi, version = version)
                
        if FLAG_BAYES_DECISION:
            BayesDecision.BayesDecision(DTR, LTR, DTE, LTE)
    
    
    
    