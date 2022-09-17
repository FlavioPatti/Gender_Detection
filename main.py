import numpy

import utilities as ut
from Graphics import graphics
from Preprocessing import pca
from Preprocessing import lda
from Generative_Models import MultivariateGaussianClassifier
from Generative_Models import NaiveBayesClassifier
from Generative_Models import TiedGaussianClassifier
from Generative_Models import TiedNaiveBayes
from Logistic_Regression import LinearLogisticRegression
from Logistic_Regression import QuadraticLogisticRegression
from Preprocessing import ZNormalization
from SVM import LinearSVM
from SVM import KernelSVM
from GMM import gmm
from Bayes_Decision_Model_Evaluation import BayesDecision
from Preprocessing import Gaussianization


SHOW_FIGURES_INIT = 0
SHOW_FIGURES_END = 0

TRAINING= 1
TESTING= 0

CALIBRATION=0
BALANCING=0
FUSION=0

SINGLEFOLD= 0
KFOLD=1

GAUSSIANIZATION= 0
ZNORMALIZATION=0
PCA = 1
LDA = 0

MVG = 1
NAIVE=1
MVG_TIED = 1
NAIVE_TIED =1

LIN_LOGISTIC = 0
QUAD_LOGISTIC = 0

LIN_SVM= 0
POL_SVM=0
RBF_SVM=0

FULL_GMM= 0
DIAG_GMM= 0
TIED_GMM= 0

def k_fold(D, L, K, algorithm, params=None, seed=0):
    """ Implementation of the k-fold cross validation approach
        D is the dataset, L the labels, K the number of folds
        algorithm is the algorithm used as classifier
        params are optional additional parameters like hyperparameters
        seed is set to 0 and it's used to randomize partitions
        return: llr and labels
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

        if GAUSSIANIZATION:
            DTE= Gaussianization.Gaussianization(DTR,DTE)
            DTR = Gaussianization.Gaussianization(DTR,DTR)
            print("Gaussianization")

        if ZNORMALIZATION:
                DTE = ZNormalization.ZNormalization(DTR,DTE)
                DTR = ZNormalization.ZNormalization(DTR,DTR)
                print("Z-normalization")
            
        if PCA:
            DTE=pca.PCA(DTR, DTE, 10)
            DTR=pca.PCA(DTR, DTR, 10)
            print("PCA dimensionality: ",DTR.shape)

        if LDA:
            DTE=lda.LDA(DTR, LTR,DTE, 1)
            DTR=lda.LDA(DTR, LTR,DTR, 1)
            print("LDA dimensionality: ",DTR.shape)
            
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
    
    return all_llr, all_labels

if __name__ == '__main__':

    #load data
    DTR, LTR, DTE, LTE = ut.load_train_and_test()
    print("Total sample of training set: ", DTR.shape[1] )
    print("Total sample of test set: ", DTE.shape[1] )
    
    print("Total sample of class 0 for training set: ", (LTR==0).sum())
    print("Total sample of class 1 for training set: ", (LTR==1).sum())
    
    print("Total sample of class 0 for test set: ", (LTE==0).sum())
    print("Total sample of class 1 for test set: ", (LTE==1).sum())
    
    hFea = {
        0: 'Feature 1',
        1: 'Feature 2',
        2: 'Feature 3',
        3: 'Feature 4',
        4: 'Feature 5',
        5: 'Feature 6',
        6: 'Feature 7',
        7: 'Feature 8',
        8: 'Feature 9',
        9: 'Feature 10',
        10: 'Feature 11',
        11: 'Feature 12'
        }
    
    DTR0=DTR[:,LTR==0]
    DTR1=DTR[:,LTR==1]
    
    # 3 applications: the main balanced and the other two unbalanced 
    applications = [[0.5,1,1],[0.1,1,1],[0.9,1,1]]

    """Flag useful to generate graphics of the various attributes of our data set"""
    if SHOW_FIGURES_INIT:
        
        """plot histograms"""
        graphics.plot_hist(DTR, LTR, hFea)
        
        """plot scatters"""
        graphics.plot_scatter(DTR, LTR, hFea)
        
        """Gaussianization of the data"""
        DTR_gauss = Gaussianization.Gaussianization(DTR, DTR)
        DTE_gauss= Gaussianization.Gaussianization(DTR, DTE)
        
        """plot gaussianization histograms"""
        graphics.plot_hist(DTR_gauss, LTR, hFea, "DTR_gauss")
        graphics.plot_hist(DTE_gauss, LTE, hFea, "DTE_gauss")
        
        """Znormalization of the data"""
        DTR_znorm = ZNormalization.ZNormalization(DTR, DTR)
        DTE_znorm = ZNormalization.ZNormalization(DTR, DTE)
        
        """plot znormalization histograms"""
        graphics.plot_hist(DTR_znorm, LTR, hFea, "DTR_znorm")
        graphics.plot_hist(DTE_znorm, LTE, hFea, "DTE_znorm")
        
        """plot correlations"""
        graphics.plot_heatmap(DTR0, "male_class_correlation")
        graphics.plot_heatmap(DTR1, "female_class_correlation")
        graphics.plot_heatmap(DTR, "global_correlation")

    if TRAINING:
        print("training")
        lambda_list = [0, 1e-6, 1e-3,1]
        if SHOW_FIGURES_END:
            listMinDCF=[]
        K=3
        if KFOLD:
            print("k-fold")
            sample_class0 = (LTR==0).sum()
            print("Sample of class 0: ", sample_class0)
            sample_class1 = (LTR==1).sum()
            print("Sample of class 1: ", sample_class1, "\n")

            for app in applications:
                pi1, Cfn, Cfp = app
                print("Application: pi1 = %.1f, Cfn = %d, Cfp = %d" %(pi1, Cfn,Cfp))
            
                if MVG:
                    print("mvg")
                    all_llrs, all_labels = k_fold(DTR, LTR, K, MultivariateGaussianClassifier.MultivariateGaussianClassifier)
                    DCF_min =  BayesDecision.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)
                    
                if NAIVE:
                    print("naive")
                    all_llrs, all_labels = k_fold(DTR, LTR, K, NaiveBayesClassifier.NaiveBayesClassifier)
                    DCF_min =  BayesDecision.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)

                if MVG_TIED:
                    print("tied gaussian")
                    all_llrs, all_labels = k_fold(DTR, LTR, K, TiedGaussianClassifier.TiedGaussianClassifier)
                    DCF_min =  BayesDecision.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)

                if NAIVE_TIED:            
                    print("tied naive")
                    all_llrs, all_labels = k_fold(DTR, LTR, K, TiedNaiveBayes.TiedNaiveBayes)
                    DCF_min =  BayesDecision.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)

                if LIN_LOGISTIC:
                    for l in lambda_list:
                    
                        print(" linear logistic regression with lamb ", l)
                        test_llrs = LinearLogisticRegression.LinearLogisticRegression(DTR, LTR, DTE, l)
                        DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                        DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                        print("DCF min= ", DCF_min)
                        print("DCF act = ", DCF_act)
                       # listMinDCF.append(DCF_min)
                        
                        if BALANCING:
                            print(" balancing of the linear logistic regression with lamb ", l)
                            test_llrs = LinearLogisticRegression.LinearLogisticRegression(DTR, LTR, DTE, l, True, 0.5)
                            DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                            DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                            print("DCF min= ", DCF_min)
                            print("DCF act = ", DCF_act)
                            #listMinDCF.append(DCF_min)
                            
                if QUAD_LOGISTIC:
                    for l in lambda_list:

                        print(" quadratic logistic regression with lamb ", l)
                        test_llrs = QuadraticLogisticRegression.QuadraticLogisticRegression(DTR, LTR, DTE, l)
                        DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                        DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                        print("DCF min= ", DCF_min)
                        print("DCF act = ", DCF_act)
                       # listMinDCF.append(DCF_min)
                        
                        """
                       #plot bayes error
                        p = numpy.linspace(-3,3,21)
                        plt.plot(p, graphics.bayes_error_plot(p, test_llrs, LTE,minCost=False), color='r')
                        plt.plot(p, graphics.bayes_error_plot(p, test_llrs, LTE,minCost=True), color='b')
                        plt.ylim([0, 1.1])
                        plt.xlim([-3, 3])
                        plt.savefig('Graphics/Generated_figures/DCFPlots/Quadratic LR-minDCF-actDCF.jpg')
                        plt.show()
                        """
                                
                        if BALANCING:
                            print(" balancing of the quadratic logistic regression with lamb ", l)
                            test_llrs = QuadraticLogisticRegression.QuadraticLogisticRegression(DTR, LTR, DTE, l,True,0.5)
                            DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                            DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                            print("DCF min= ", DCF_min)
                            print("DCF act = ", DCF_act)
                           # listMinDCF.append(DCF_min)
                        
                                
                if LIN_SVM:
                    
                    K_list = [1, 10]
                    C_list = [1e-3, 0.1, 1]
                    for K_ in K_list:
                        for C in C_list:
                            
                            print("SVM Linear: K = %f, C = %f" % (K_,C), "\n")
                            test_llrs = LinearSVM.train_SVM_linear(DTR, LTR, DTE, C, K_, 0, balanced = False)  
                            DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                            DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                            print("DCF min= ", DCF_min)
                            print("DCF act = ", DCF_act)
                            
                            if BALANCING:
                                print("SVM Linear with balancing: K = %f, C = %f" % (K_,C), "\n")
                                test_llrs = LinearSVM.train_SVM_linear(DTR, LTR, DTE, C, K_, 0.5, balanced= True)  
                                DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                                DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                                print("DCF min= ", DCF_min)
                                print("DCF act = ", DCF_act)

                if POL_SVM:        
                    K_list = [1]
                    C_list = [1e-5,1e-4,1e-3,1e-2,1e-1]       
                    c_list = [0,1,10,30]
                    for K_ in K_list:
                        for c in c_list:  
                            for C in C_list:
                                
                                print("SVM Polynomial Kernel: K = %f, C = %f, d=2, c= %f" % (K_,C,c), "\n")
                                test_llrs = KernelSVM.quad_kernel_svm(DTR, LTR, DTE, C, c, 0, K_, "poly", balanced = False)
                                DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                                DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                                print("DCF min= ", DCF_min)
                                print("DCF act = ", DCF_act)
                                
                                if BALANCING:
                                    print("SVM Polynomial Kernel with balancing: K = %f, C = %f, d=2, c= %f" % (K_,C,c), "\n")
                                    test_llrs = KernelSVM.quad_kernel_svm(DTR, LTR, DTE, C, c, 0, K_, "poly", balanced = True)
                                    DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                                    DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                                    print("DCF min= ", DCF_min)
                                    print("DCF act = ", DCF_act)
                                    listMinDCF.append(DCF_min)   
                     
                if RBF_SVM:       
                    K_list = [1]
                    C_list = [0.001,0.01,0.1,1,10,100,1000]                 
                    g_list = [1e-5,1e-4,1e-3,1e-2]
                    for K_ in K_list:
                        for g in g_list:  
                            for C in C_list:   
                                
                                print("SVM RBF Kernel: K = %f, C = %f, g=%f" % (K_,C,g), "\n")
                                test_llrs = KernelSVM.quad_kernel_svm(DTR, LTR, DTE, C, 0, g, K_, "RBF", balanced = False)
                                DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                                DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                                print("DCF min= ", DCF_min)
                                print("DCF act = ", DCF_act)
                                #listMinDCF.append(DCF_min)
                                
                                if BALANCING:
                                    print("SVM RBF Kernel with balancing: K = %f, C = %f, g=%f" % (K_,C,g), "\n")
                                    test_llrs1 = KernelSVM.quad_kernel_svm(DTR, LTR, DTE, C, 0, g, K_, "RBF", balanced = True)
                                    DCF_min =  BayesDecision.compute_min_DCF(test_llrs1, LTE, pi1, Cfn, Cfp)
                                    DCF_act = BayesDecision.compute_act_DCF(test_llrs1, LTE, pi1, Cfn, Cfp)
                                    print("DCF min= ", DCF_min)
                                    print("DCF act = ", DCF_act)
                                    listMinDCF.append(DCF_min)
                        
                if FULL_GMM:
                    psi = 0.01
                    M_list = [2,4,8] 
                    for M in M_list:
                        
                        print("GMM version = %s, M = %d, psi = %f" % ("full", M, psi), "\n")
                        test_llrs= gmm.GMM_classifier(DTR, LTR, DTE, M, psi, "full") 
                        DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                        DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                        print("DCF min= ", DCF_min)
                        print("DCF act = ", DCF_act)  

                if DIAG_GMM:
                    psi = 0.01
                    M_list = [2,4,8] 
                    for M in M_list:
                        
                        print("GMM version = %s, M = %d, psi = %f" % ("diagonal", M, psi), "\n")
                        test_llrs= gmm.GMM_classifier(DTR, LTR, DTE, M, psi, "diagonal") 
                        DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                        DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                        print("DCF min= ", DCF_min)
                        print("DCF act = ", DCF_act)          
                
                if TIED_GMM:
                    psi = 0.01
                    M_list = [2,4,8] 
                    for M in M_list:
                        
                        print("GMM version = %s, M = %d, psi = %f" % ("tied", M, psi), "\n")
                        test_llrs= gmm.GMM_classifier(DTR, LTR, DTE, M, psi, "tied") 
                        DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                        DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                        print("DCF min= ", DCF_min)
                        print("DCF act = ", DCF_act)
                     
                """Fusion of our best models: Balanced SVM Linear K=1, C=0.1 with PCA=7 and Z-Normalized Quadratic LR with lambda=1e-3"""     
                """
                if FUSION:
                    #First model
                    
                    DTE1=pca.PCA(DTR, DTE, 7)
                    DTR_V1=pca.PCA(DTR, DTR_V, 7)
                    DTR1=pca.PCA(DTR, DTR, 7)
                    print("PCA dimensionality: ",DTR.shape)
                    
                    print("SVM Linear with balancing: K = %f, C = %f" % (1,0.1), "\n")
                    test_llrs1 = LinearSVM.train_SVM_linear(DTR1, LTR, DTE1, 0.1, 1, 0.5, balanced= True)  
                    DCF_min =  BayesDecision.compute_min_DCF(test_llrs1, LTE, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(test_llrs1, LTE, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)
                        
                    #Second model
                    
                    DTE2 = ZNormalization.ZNormalization(DTR,DTE)
                    DTR_V2 = ZNormalization.ZNormalization(DTR,DTR_V)
                    DTR2 = ZNormalization.ZNormalization(DTR,DTR)
                    print("Z-normalization")  
                    
                    print(" quadratic logistic regression with lamb ", 1e-3)
                    test_llrs2 = QuadraticLogisticRegression.QuadraticLogisticRegression(DTR2, LTR, DTE2, 1e-3)
                    DCF_min =  BayesDecision.compute_min_DCF(test_llrs2, LTE, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(test_llrs2, LTE, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)
                    
                    
                    test_llrs_stacked=numpy.vstack((test_llrs1,test_llrs2))
                    print(test_llrs_stacked.shape)
                    
                    tra_llrs1 = LinearSVM.train_SVM_linear(DTR1, LTR, DTR_V1, 0.1, 1, 0.5, balanced= True) 
                    tra_llrs2 = QuadraticLogisticRegression.QuadraticLogisticRegression(DTR2, LTR, DTR_V2, 1e-3) 
               
                    tra_llrs_stacked=numpy.vstack((tra_llrs1,tra_llrs2))
                    print(tra_llrs_stacked.shape)
                    print("fusion")
                    fus_llrs= LinearLogisticRegression.BalancedLinearLogisticRegression(tra_llrs_stacked,LTR_V,test_llrs_stacked, 1e-03,0.5)
                    print(fus_llrs.shape)
                    DCF_min =  BayesDecision.compute_min_DCF(fus_llrs, LTE, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(fus_llrs, LTE, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)
                    """  
              
            """plot of the minDCF at the end of the computation"""
            """
            if SHOW_FIGURES_END:
                lambda_list_plot = [1e-12, 1e-6, 1e-3,1]
                print("listMinDCF lenght: ", len(listMinDCF))
                graphics.plotDCFprior(lambda_list_plot,listMinDCF,"lambda Balanced Linear LR")
            """
            """
            if SHOW_FIGURES_END:
                C_list_plot=[1e-5,1e-4,1e-3,1e-2,1e-1]
                print("listMinDCF lenght: ", len(listMinDCF))
                graphics.plotDCFc(C_list_plot,listMinDCF,"C Balanced SVM polynomial kernel")
            """
            """
            if SHOW_FIGURES_END:
                C_list_plot=[0.001,0.01,0.1,1,10,100,1000]
                print("listMinDCF lenght: ", len(listMinDCF))
                graphics.plotDCFg(C_list_plot,listMinDCF,"C Balanced SVM RBF kernel")
            """
            
    if TESTING:
        print("testing")
        lambda_list = [0, 1e-6, 1e-3, 1]
        listMinDCF=[]
        
        """ we performed testing using 1/2 of the original training set for training, and the evaluation/test set for evaluating our choices"""
        if SINGLEFOLD:
            print("Singlefold")
            (DTR_TRA, LTR_TRA), (DTR_V,LTR_V) = ut.split_db_2to1(DTR, LTR,2/3)
            sample_class0 = (LTR==0).sum()
            print("Sample of class 0: ", sample_class0)
            sample_class1 = (LTR==1).sum()
            print("Sample of class 1: ", sample_class1, "\n")
            