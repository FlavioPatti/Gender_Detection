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


SHOW_FIGURES_INIT = False
SHOW_FIGURES_END = False

TRAINING = False
TESTING = False

CALIBRATION = False
BALANCING = False
FUSION = False

GAUSSIANIZATION = False
ZNORMALIZATION = False
PCA = False
LDA = False

MVG = False
NAIVE = False
MVG_TIED = False
NAIVE_TIED = False

LIN_LOGISTIC = False
QUAD_LOGISTIC = False

LIN_SVM = False
POL_SVM = False
RBF_SVM = False

FULL_GMM = False
DIAG_GMM = False
TIED_GMM = False

def k_fold(D, L, K, algorithm, params=None, params_cal=None, seed=0):
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

    all_llrs = []
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

        #apply preprocessing to data
        if GAUSSIANIZATION:
            DTE= Gaussianization.Gaussianization(DTR,DTE)
            DTR = Gaussianization.Gaussianization(DTR,DTR)
            print("Gaussianization")

        if ZNORMALIZATION:
                DTE = ZNormalization.ZNormalization(DTR,DTE)
                DTR = ZNormalization.ZNormalization(DTR,DTR)
                print("Z-normalization")
            
        if PCA:
            m_pca = 11
            DTE=pca.PCA(DTR, DTE, m_pca)
            DTR=pca.PCA(DTR, DTR, m_pca)
            print("PCA dimensionality: ",DTR.shape)

        if LDA:
            m_lda = 1
            DTE=lda.LDA(DTR, LTR,DTE, m_lda)
            DTR=lda.LDA(DTR, LTR,DTR, m_lda)
            print("LDA dimensionality: ",DTR.shape)
            
        # calculate scores
        if params is not None:
            llr = algorithm(DTR, LTR, DTE, *params)
        else:
            llr = algorithm(DTR, LTR, DTE)
        # add scores and labels for this fold in total
        all_llrs.append(llr)
        all_labels.append(LTE)

    all_llrs = numpy.hstack(all_llrs)
    all_labels = numpy.hstack(all_labels)

    if params_cal:
        llr_cal = []
        labels_cal = []
        idx_numbers = numpy.arange(all_llrs.size)
        idx_partitions = []
        for i in range(0, all_llrs.size, sizePartitions):
            idx_partitions.append(list(idx_numbers[i:i+sizePartitions]))
        for i in range(K):

            idx_test = idx_partitions[i]
            idx_train = idx_partitions[0:i] + idx_partitions[i+1:]

            # from lists of lists collapse the elemnts in a single list
            idx_train = sum(idx_train, [])

            # partition the data and labels using the already partitioned indexes
            STR = all_llrs[idx_train]
            STE = all_llrs[idx_test]
            LTR = all_labels[idx_train]
            LTE = all_labels[idx_test]
            
            cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(STR,LTR,STE,params_cal,0.5)
            llr_cal.append(cal_llrs)
            labels_cal.append(LTE)

        llr_cal = numpy.hstack(llr_cal)
        labels_cal = numpy.hstack(labels_cal)

        return llr_cal, labels_cal
    
    return all_llrs, all_labels

if __name__ == '__main__':
        
    #load data
    DTR, LTR, DTE, LTE = ut.load_train_and_test()
    
    #print stats
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
    applications = [[0.5,1,1] , [0.1,1,1], [0.9,1,1]]

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
        lambda_list = [0, 1e-6, 1e-3, 1]
        
        if SHOW_FIGURES_END:
            listMinDCF=[]
            
        """ We performed training using a k-fold approach with k=3"""
        K=3
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
                
                if CALIBRATION:
                    for l in lambda_list:
                        print(" calibration with logistic regression with lamb ", l)
                        all_llrs, all_labels = k_fold(DTR, LTR, K, MultivariateGaussianClassifier.MultivariateGaussianClassifier, None, l)
                        DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                        print("DCF calibrated act = ", DCF_act)
            if NAIVE:
                print("naive")
                all_llrs, all_labels = k_fold(DTR, LTR, K, NaiveBayesClassifier.NaiveBayesClassifier)
                DCF_min =  BayesDecision.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                print("DCF min= ", DCF_min)
                print("DCF act = ", DCF_act)
                
                if CALIBRATION:
                    for l in lambda_list:
                        print(" calibration with logistic regression with lamb ", l)
                        all_llrs, all_labels = k_fold(DTR, LTR, K, NaiveBayesClassifier.NaiveBayesClassifier, None, l)
                        DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                        print("DCF calibrated act = ", DCF_act)

            if MVG_TIED:
                print("tied gaussian")
                all_llrs, all_labels = k_fold(DTR, LTR, K, TiedGaussianClassifier.TiedGaussianClassifier)
                DCF_min =  BayesDecision.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                print("DCF min= ", DCF_min)
                print("DCF act = ", DCF_act)
                
                if CALIBRATION:
                    for l in lambda_list:
                        print(" calibration with logistic regression with lamb ", l)
                        all_llrs, all_labels = k_fold(DTR, LTR, K, TiedGaussianClassifier.TiedGaussianClassifier, None, l)
                        DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                        print("DCF calibrated act = ", DCF_act)

            if NAIVE_TIED:            
                print("tied naive")
                all_llrs, all_labels = k_fold(DTR, LTR, K, TiedNaiveBayes.TiedNaiveBayes)
                DCF_min =  BayesDecision.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                print("DCF min= ", DCF_min)
                print("DCF act = ", DCF_act)  
                
                if CALIBRATION:
                    for l in lambda_list:
                        print(" calibration with logistic regression with lamb ", l)
                        all_llrs, all_labels = k_fold(DTR, LTR, K, TiedNaiveBayes.TiedNaiveBayes, None, l)
                        DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                        print("DCF calibrated act = ", DCF_act)

            if LIN_LOGISTIC:
                for l in lambda_list:
                    
                    print(" linear logistic regression with lamb ", l)
                    all_llrs, all_labels = k_fold(DTR, LTR, K, LinearLogisticRegression.LinearLogisticRegression, [l])
                    DCF_min =  BayesDecision.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)
                    #listMinDCF.append(DCF_min)
                    
                    if CALIBRATION:
                        for l2 in lambda_list:
                            print(" calibration with logistic regression with lamb ", l2)
                            all_llrs, all_labels = k_fold(DTR, LTR, K, LinearLogisticRegression.LinearLogisticRegression, [l], l2)
                            DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                            print("DCF calibrated act = ", DCF_act)
                    
                    if BALANCING:
                        print(" balancing of the linear logistic regression with lamb ", l)
                        all_llrs, all_labels = k_fold(DTR, LTR, K, LinearLogisticRegression.LinearLogisticRegression, [l, True, 0.5])
                        DCF_min =  BayesDecision.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                        DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                        print("DCF min= ", DCF_min)
                        print("DCF act = ", DCF_act)
                        #listMinDCF.append(DCF_min)
                        
            if QUAD_LOGISTIC:
                for l in lambda_list:

                    print(" quadratic logistic regression with lamb ", l)
                    all_llrs, all_labels = k_fold(DTR, LTR, K, QuadraticLogisticRegression.QuadraticLogisticRegression, [l])
                    DCF_min =  BayesDecision.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
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
                    
                    if CALIBRATION:
                        for l2 in lambda_list:
                            print(" calibration with logistic regression with lamb ", l2)
                            all_llrs, all_labels = k_fold(DTR, LTR, K, QuadraticLogisticRegression.QuadraticLogisticRegression, [l], l2)
                            DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                            print("DCF calibrated act = ", DCF_act)
                            
                    if BALANCING:
                        print(" balancing of the quadratic logistic regression with lamb ", l)
                        all_llrs, all_labels = k_fold(DTR, LTR, K, QuadraticLogisticRegression.QuadraticLogisticRegression, [l, True, 0.5])
                        DCF_min =  BayesDecision.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                        DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                        print("DCF min= ", DCF_min)
                        print("DCF act = ", DCF_act)
                        # listMinDCF.append(DCF_min)
                    
                            
            if LIN_SVM:
                
                K_list = [1]
                C_list = [1e-3, 1e-2, 1e-1, 1]
                for K_ in K_list:
                    for C in C_list:
                        
                        print("SVM Linear: K = %f, C = %f" % (K_,C), "\n")
                        all_llrs, all_labels = k_fold(DTR, LTR, K, LinearSVM.train_SVM_linear, [C, K_])  
                        DCF_min =  BayesDecision.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                        DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                        print("DCF min= ", DCF_min)
                        print("DCF act = ", DCF_act)
                        
                        if CALIBRATION:
                            for l in lambda_list:
                                print(" calibration with logistic regression with lamb ", l)
                                all_llrs, all_labels = k_fold(DTR, LTR, K, LinearSVM.train_SVM_linear, [C, K_], l)
                                DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                                print("DCF calibrated act = ", DCF_act)
                        
                        if BALANCING:
                            print("SVM Linear with balancing: K = %f, C = %f" % (K_,C), "\n")
                            all_llrs, all_labels = k_fold(DTR, LTR, K, LinearSVM.train_SVM_linear, [C, K_,True,0.5])  
                            DCF_min =  BayesDecision.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                            DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                            print("DCF min= ", DCF_min)
                            print("DCF act = ", DCF_act)

            if POL_SVM:        
                K_list = [1]
                C_list = [1e-5,1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]       
                c_list = [0,1,10,30]
                for K_ in K_list:
                    for c in c_list:  
                        for C in C_list:
                            
                            print("SVM Polynomial Kernel: K = %f, C = %f, d=2, c= %f" % (K_,C,c), "\n")
                            all_llrs, all_labels = k_fold(DTR, LTR, K, KernelSVM.kernel_svm, [C, c,None,K_, "poly"]) 
                            DCF_min =  BayesDecision.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                            DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                            print("DCF min= ", DCF_min)
                            print("DCF act = ", DCF_act)
                            
                            if CALIBRATION:
                                for l in lambda_list:
                                    print(" calibration with logistic regression with lamb ", l)
                                    all_llrs, all_labels = k_fold(DTR, LTR, K, KernelSVM.kernel_svm, [C, c, None, K_, "poly" ], l)
                                    DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                                    print("DCF calibrated act = ", DCF_act)
                            
                            if BALANCING:
                                print("SVM Polynomial Kernel with balancing: K = %f, C = %f, d=2, c= %f" % (K_,C,c), "\n")
                                all_llrs, all_labels = k_fold(DTR, LTR, K, KernelSVM.kernel_svm, [C, c,None,K_, "poly", True,0.5]) 
                                DCF_min =  BayesDecision.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                                DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                                print("DCF min= ", DCF_min)
                                print("DCF act = ", DCF_act)
                                listMinDCF.append(DCF_min)   
                    
            if RBF_SVM:       
                K_list = [1]
                C_list = [1e-5,1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]                 
                g_list = [1e-5,1e-4,1e-3,1e-2]
                for K_ in K_list:
                    for g in g_list:  
                        for C in C_list:   
                            
                            print("SVM RBF Kernel: K = %f, C = %f, g=%f" % (K_,C,g), "\n")
                            all_llrs, all_labels = k_fold(DTR, LTR, K, KernelSVM.kernel_svm, [C, None,g, K_, "RBF"]) 
                            DCF_min =  BayesDecision.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                            DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                            print("DCF min= ", DCF_min)
                            print("DCF act = ", DCF_act)
                            #listMinDCF.append(DCF_min)
                            
                            if CALIBRATION:
                                for l in lambda_list:
                                    print(" calibration with logistic regression with lamb ", l)
                                    all_llrs, all_labels = k_fold(DTR, LTR, K, KernelSVM.kernel_svm, [C, None, g, K_, "RBF"], l)
                                    DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                                    print("DCF calibrated act = ", DCF_act)
                            
                            if BALANCING:
                                print("SVM RBF Kernel with balancing: K = %f, C = %f, g=%f" % (K_,C,g), "\n")
                                all_llrs, all_labels = k_fold(DTR, LTR, K, KernelSVM.kernel_svm, [C, None,g, K_, "RBF", True, 0.5])
                                DCF_min =  BayesDecision.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                                DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                                print("DCF min= ", DCF_min)
                                print("DCF act = ", DCF_act)
                                listMinDCF.append(DCF_min)
                    
            if FULL_GMM:
                psi = 0.01
                M_list = [2,4,8] 
                for M in M_list:
                    
                    print("GMM version = %s, M = %d, psi = %f" % ("full", M, psi), "\n")
                    all_llrs, all_labels = k_fold(DTR, LTR, K, gmm.GMM_classifier, [M,psi,"full"]) 
                    DCF_min =  BayesDecision.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)  
                    
                    if CALIBRATION:
                        for l in lambda_list:
                            print(" calibration with logistic regression with lamb ", l)
                            all_llrs, all_labels = k_fold(DTR, LTR, K, gmm.GMM_classifier, [M, psi, "full"], l)
                            DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                            print("DCF calibrated act = ", DCF_act)

            if DIAG_GMM:
                psi = 0.01
                M_list = [2,4,8] 
                for M in M_list:
                    
                    print("GMM version = %s, M = %d, psi = %f" % ("diagonal", M, psi), "\n")
                    all_llrs, all_labels = k_fold(DTR, LTR, K, gmm.GMM_classifier, [M,psi,"diagonal"])  
                    DCF_min =  BayesDecision.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act) 
                    
                    if CALIBRATION:
                        for l in lambda_list:
                            print(" calibration with logistic regression with lamb ", l)
                            all_llrs, all_labels = k_fold(DTR, LTR, K, gmm.GMM_classifier, [M, psi, "diagonal"], l)
                            DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                            print("DCF calibrated act = ", DCF_act)         
            
            if TIED_GMM:
                psi = 0.01
                M_list = [2,4,8] 
                for M in M_list:
                    
                    print("GMM version = %s, M = %d, psi = %f" % ("tied", M, psi), "\n")
                    all_llrs, all_labels = k_fold(DTR, LTR, K, gmm.GMM_classifier, [M,psi,"tied"])  
                    DCF_min =  BayesDecision.compute_min_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)
                    
                    if CALIBRATION:
                        for l in lambda_list:
                            print(" calibration with logistic regression with lamb ", l)
                            all_llrs, all_labels = k_fold(DTR, LTR, K, gmm.GMM_classifier, [M, psi, "tied"], l)
                            DCF_act = BayesDecision.compute_act_DCF(all_llrs, all_labels, pi1, Cfn, Cfp)
                            print("DCF calibrated act = ", DCF_act)
                    
            """Fusion of our best models: Tied GMM with M=8 and SVM RBF with C=1 g=1e-3"""     
            
            if FUSION:
                #Tied GMM m=8 0.031
                #RBF C=1 g=1e-3 0.039
                
                #First model
                psi = 0.01
                M=8
                print("GMM version = %s, M = %d, psi = %f" % ("tied", M, psi), "\n")
                all_llrs1, all_labels = k_fold(DTR, LTR, K, gmm.GMM_classifier, [M,psi,"tied"])  
                DCF_min =  BayesDecision.compute_min_DCF(all_llrs1, all_labels, pi1, Cfn, Cfp)
                DCF_act = BayesDecision.compute_act_DCF(all_llrs1, all_labels, pi1, Cfn, Cfp)
                print("DCF min= ", DCF_min)
                print("DCF act = ", DCF_act)
                    
                #Second model  
                K_ = 1
                C = 1                 
                g = 1e-3  
                print("SVM RBF Kernel: K = %f, C = %f, g=%f" % (K_,C,g), "\n")
                all_llrs2, all_labels = k_fold(DTR, LTR, K, KernelSVM.kernel_svm, [C, None,g, K_, "RBF"]) 
                DCF_min =  BayesDecision.compute_min_DCF(all_llrs2, all_labels, pi1, Cfn, Cfp)
                DCF_act = BayesDecision.compute_act_DCF(all_llrs2, all_labels, pi1, Cfn, Cfp)
                print("DCF min= ", DCF_min)
                print("DCF act = ", DCF_act)
                
                
                test_llrs_stacked=numpy.vstack((all_llrs1,all_llrs2))
                print(test_llrs_stacked.shape)

                llr_fus = []
                labels_fus = []
                idx_numbers = numpy.arange(test_llrs_stacked.shape[1])
                idx_partitions = []
                sizePartitions = int(test_llrs_stacked.shape[1]/K)
                for i in range(0, test_llrs_stacked.shape[1], sizePartitions):
                    idx_partitions.append(list(idx_numbers[i:i+sizePartitions]))
                for i in range(K):

                    idx_test = idx_partitions[i]
                    idx_train = idx_partitions[0:i] + idx_partitions[i+1:]

                    # from lists of lists collapse the elemnts in a single list
                    idx_train = sum(idx_train, [])

                    # partition the data and labels using the already partitioned indexes
                    STR = test_llrs_stacked[:,idx_train]
                    STE = test_llrs_stacked[:,idx_test]
                    LTRS = all_labels[idx_train]
                    LTES = all_labels[idx_test]
                    
                    print("fusion")
                    fus_llr= LinearLogisticRegression.LinearLogisticRegression(STR,LTRS, STE, 1e-03, True, 0.5)
                    print(fus_llr.shape)
                    llr_fus.append(fus_llr)
                    labels_fus.append(LTES)

                llr_fus = numpy.hstack(llr_fus)
                labels_fus = numpy.hstack(labels_fus)

                DCF_min =  BayesDecision.compute_min_DCF(llr_fus, labels_fus, pi1, Cfn, Cfp)
                DCF_act = BayesDecision.compute_act_DCF(llr_fus, labels_fus, pi1, Cfn, Cfp)
                print("DCF min= ", DCF_min)
                print("DCF act = ", DCF_act)
                  
            
        """plot of the minDCF at the end of the computation"""
        """
        if SHOW_FIGURES_END:
            lambda_list_plot = [1e-12, 1e-6, 1e-3,1]
            print("listMinDCF lenght: ", len(listMinDCF))
            graphics.plotDCFprior(lambda_list_plot,listMinDCF,"lambda Quadratic LR - Raw")
        """
        """
        if SHOW_FIGURES_END:
            C_list_plot = [1e-3,1e-2, 1e-1, 1]
            print("listMinDCF lenght: ", len(listMinDCF))
            graphics.plotDCFprior(C_list_plot,listMinDCF,"C Linear SVM - Raw")
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
        
        #load data
        DTR, LTR, DTE, LTE = ut.load_train_and_test()
         
        #print stats
        sample_class0 = (LTR==0).sum()
        print("Sample of class 0: ", sample_class0)
        sample_class1 = (LTR==1).sum()
        print("Sample of class 1: ", sample_class1, "\n")

        #apply preprocessing to data
        if GAUSSIANIZATION:
            DTE= Gaussianization.Gaussianization(DTR,DTE)
            DTR = Gaussianization.Gaussianization(DTR,DTR)
            print("Gaussianization")

        if ZNORMALIZATION:
            DTE = ZNormalization.ZNormalization(DTR,DTE)
            DTR = ZNormalization.ZNormalization(DTR,DTR)
            print("Z-normalization")   
                
        if PCA:
            m_pca = 11
            DTE=pca.PCA(DTR, DTE, m_pca)
            DTR=pca.PCA(DTR, DTR, m_pca)
            print("PCA dimensionality: ",DTR.shape)

        if LDA:
            m_lda = 1
            DTE=lda.LDA(DTR, LTR,DTE, m_lda)
            DTR=lda.LDA(DTR, LTR,DTR, m_lda)
            print("LDA dimensionality: ",DTR.shape)

        for app in applications:
            pi1, Cfn, Cfp = app
            print("Application: pi1 = %.1f, Cfn = %d, Cfp = %d" %(pi1, Cfn,Cfp))
            
            if MVG:
                print("mvg")
                test_llrs = MultivariateGaussianClassifier.MultivariateGaussianClassifier(DTR, LTR, DTE)
                DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                print("DCF min= ", DCF_min)
                print("DCF act = ", DCF_act)

                if CALIBRATION:
                    for l in lambda_list:
                        print(" calibration with logistic regression with lamb ", l)
                        K=3
                        sizePartitions=int(test_llrs.size/K)
                        llr_cal = []
                        labels_cal = []
                        idx_numbers = numpy.arange(test_llrs.size)
                        idx_partitions = []
                        for i in range(0, test_llrs.size, sizePartitions):
                            idx_partitions.append(list(idx_numbers[i:i+sizePartitions]))
                        for i in range(K):

                            idx_test = idx_partitions[i]
                            idx_train = idx_partitions[0:i] + idx_partitions[i+1:]

                            # from lists of lists collapse the elemnts in a single list
                            idx_train = sum(idx_train, [])

                            # partition the data and labels using the already partitioned indexes
                            STR = test_llrs[idx_train]
                            STE = test_llrs[idx_test]
                            LTRS = LTE[idx_train]
                            LTES = LTE[idx_test]
                            
                            cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(STR,LTRS,STE,l,0.5)
                            llr_cal.append(cal_llrs)
                            labels_cal.append(LTES)
                        llr_cal = numpy.hstack(llr_cal)
                        labels_cal = numpy.hstack(labels_cal)
                        DCF_act = BayesDecision.compute_act_DCF(llr_cal, labels_cal, pi1, Cfn, Cfp)
                        print("DCF calibrated act = ", DCF_act)
                        
            if NAIVE:
                print("naive")
                test_llrs = NaiveBayesClassifier.NaiveBayesClassifier(DTR, LTR, DTE)
                DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                print("DCF min= ", DCF_min)
                print("DCF act = ", DCF_act)
                
                if CALIBRATION:
                    for l in lambda_list:
                        print(" calibration with logistic regression with lamb ", l)
                        K=3
                        sizePartitions=int(test_llrs.size/K)
                        llr_cal = []
                        labels_cal = []
                        idx_numbers = numpy.arange(test_llrs.size)
                        idx_partitions = []
                        for i in range(0, test_llrs.size, sizePartitions):
                            idx_partitions.append(list(idx_numbers[i:i+sizePartitions]))
                        for i in range(K):

                            idx_test = idx_partitions[i]
                            idx_train = idx_partitions[0:i] + idx_partitions[i+1:]

                            # from lists of lists collapse the elemnts in a single list
                            idx_train = sum(idx_train, [])

                            # partition the data and labels using the already partitioned indexes
                            STR = test_llrs[idx_train]
                            STE = test_llrs[idx_test]
                            LTRS = LTE[idx_train]
                            LTES = LTE[idx_test]
                            
                            cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(STR,LTRS,STE,l,0.5)
                            llr_cal.append(cal_llrs)
                            labels_cal.append(LTES)
                        llr_cal = numpy.hstack(llr_cal)
                        labels_cal = numpy.hstack(labels_cal)
                        DCF_act = BayesDecision.compute_act_DCF(llr_cal, labels_cal, pi1, Cfn, Cfp)
                        print("DCF calibrated act = ", DCF_act)

            if MVG_TIED:
                print("tied gaussian")
                test_llrs = TiedGaussianClassifier.TiedGaussianClassifier(DTR, LTR, DTE)
                DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                print("DCF min= ", DCF_min)
                print("DCF act = ", DCF_act)
                
                if CALIBRATION:
                    for l in lambda_list:
                        print(" calibration with logistic regression with lamb ", l)
                        K=3
                        sizePartitions=int(test_llrs.size/K)
                        llr_cal = []
                        labels_cal = []
                        idx_numbers = numpy.arange(test_llrs.size)
                        idx_partitions = []
                        for i in range(0, test_llrs.size, sizePartitions):
                            idx_partitions.append(list(idx_numbers[i:i+sizePartitions]))
                        for i in range(K):

                            idx_test = idx_partitions[i]
                            idx_train = idx_partitions[0:i] + idx_partitions[i+1:]

                            # from lists of lists collapse the elemnts in a single list
                            idx_train = sum(idx_train, [])

                            # partition the data and labels using the already partitioned indexes
                            STR = test_llrs[idx_train]
                            STE = test_llrs[idx_test]
                            LTRS = LTE[idx_train]
                            LTES = LTE[idx_test]
                            
                            cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(STR,LTRS,STE,l,0.5)
                            llr_cal.append(cal_llrs)
                            labels_cal.append(LTES)
                        llr_cal = numpy.hstack(llr_cal)
                        labels_cal = numpy.hstack(labels_cal)
                        DCF_act = BayesDecision.compute_act_DCF(llr_cal, labels_cal, pi1, Cfn, Cfp)
                        print("DCF calibrated act = ", DCF_act)

            if NAIVE_TIED:            
                print("tied naive")
                test_llrs = TiedNaiveBayes.TiedNaiveBayes(DTR, LTR, DTE)
                DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                print("DCF min= ", DCF_min)
                print("DCF act = ", DCF_act)
                
                if CALIBRATION:
                    for l in lambda_list:
                        print(" calibration with logistic regression with lamb ", l)
                        K=3
                        sizePartitions=int(test_llrs.size/K)
                        llr_cal = []
                        labels_cal = []
                        idx_numbers = numpy.arange(test_llrs.size)
                        idx_partitions = []
                        for i in range(0, test_llrs.size, sizePartitions):
                            idx_partitions.append(list(idx_numbers[i:i+sizePartitions]))
                        for i in range(K):

                            idx_test = idx_partitions[i]
                            idx_train = idx_partitions[0:i] + idx_partitions[i+1:]

                            # from lists of lists collapse the elemnts in a single list
                            idx_train = sum(idx_train, [])

                            # partition the data and labels using the already partitioned indexes
                            STR = test_llrs[idx_train]
                            STE = test_llrs[idx_test]
                            LTRS = LTE[idx_train]
                            LTES = LTE[idx_test]
                            
                            cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(STR,LTRS,STE,l,0.5)
                            llr_cal.append(cal_llrs)
                            labels_cal.append(LTES)
                        llr_cal = numpy.hstack(llr_cal)
                        labels_cal = numpy.hstack(labels_cal)
                        DCF_act = BayesDecision.compute_act_DCF(llr_cal, labels_cal, pi1, Cfn, Cfp)
                        print("DCF calibrated act = ", DCF_act)

            if LIN_LOGISTIC:
                for l in lambda_list:
                    print(" linear logistic regression with lamb ", l)
                    test_llrs = LinearLogisticRegression.LinearLogisticRegression(DTR, LTR, DTE, l)
                    DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)
                    #listMinDCF.append(DCF_min)
                    
                if CALIBRATION:
                    for l2 in lambda_list:
                        print(" calibration with logistic regression with lamb ", l2)
                        K=3
                        sizePartitions=int(test_llrs.size/K)
                        llr_cal = []
                        labels_cal = []
                        idx_numbers = numpy.arange(test_llrs.size)
                        idx_partitions = []
                        for i in range(0, test_llrs.size, sizePartitions):
                            idx_partitions.append(list(idx_numbers[i:i+sizePartitions]))
                        for i in range(K):

                            idx_test = idx_partitions[i]
                            idx_train = idx_partitions[0:i] + idx_partitions[i+1:]

                            # from lists of lists collapse the elemnts in a single list
                            idx_train = sum(idx_train, [])

                            # partition the data and labels using the already partitioned indexes
                            STR = test_llrs[idx_train]
                            STE = test_llrs[idx_test]
                            LTRS = LTE[idx_train]
                            LTES = LTE[idx_test]
                            
                            cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(STR,LTRS,STE,l2,0.5)
                            llr_cal.append(cal_llrs)
                            labels_cal.append(LTES)
                        llr_cal = numpy.hstack(llr_cal)
                        labels_cal = numpy.hstack(labels_cal)
                        DCF_act = BayesDecision.compute_act_DCF(llr_cal, labels_cal, pi1, Cfn, Cfp)
                        print("DCF calibrated act = ", DCF_act)
                        
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
                    
                if CALIBRATION:
                    for l2 in lambda_list:
                        print(" calibration with logistic regression with lamb ", l2)
                        K=3
                        sizePartitions=int(test_llrs.size/K)
                        llr_cal = []
                        labels_cal = []
                        idx_numbers = numpy.arange(test_llrs.size)
                        idx_partitions = []
                        for i in range(0, test_llrs.size, sizePartitions):
                            idx_partitions.append(list(idx_numbers[i:i+sizePartitions]))
                        for i in range(K):

                            idx_test = idx_partitions[i]
                            idx_train = idx_partitions[0:i] + idx_partitions[i+1:]

                            # from lists of lists collapse the elemnts in a single list
                            idx_train = sum(idx_train, [])

                            # partition the data and labels using the already partitioned indexes
                            STR = test_llrs[idx_train]
                            STE = test_llrs[idx_test]
                            LTRS = LTE[idx_train]
                            LTES = LTE[idx_test]
                            
                            cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(STR,LTRS,STE,l2,0.5)
                            llr_cal.append(cal_llrs)
                            labels_cal.append(LTES)
                        llr_cal = numpy.hstack(llr_cal)
                        labels_cal = numpy.hstack(labels_cal)
                        DCF_act = BayesDecision.compute_act_DCF(llr_cal, labels_cal, pi1, Cfn, Cfp)
                        print("DCF calibrated act = ", DCF_act)
                    
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
                    test_llrs = QuadraticLogisticRegression.QuadraticLogisticRegression(DTR, LTR, DTE, l, True, 0.5)
                    DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)
                    # listMinDCF.append(DCF_min)
                    
                            
            if LIN_SVM:
                
                K_list = [1]
                C_list = [1e-3,1e-2, 1e-1, 1]
                for K_ in K_list:
                    for C in C_list:
                        
                        print("SVM Linear: K = %f, C = %f" % (K_,C), "\n")
                        test_llrs = LinearSVM.train_SVM_linear(DTR, LTR, DTE, C, K_)  
                        DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                        DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                        print("DCF min= ", DCF_min)
                        print("DCF act = ", DCF_act)
                        
                if CALIBRATION:
                    for l in lambda_list:
                        print(" calibration with logistic regression with lamb ", l)
                        K=3
                        sizePartitions=int(test_llrs.size/K)
                        llr_cal = []
                        labels_cal = []
                        idx_numbers = numpy.arange(test_llrs.size)
                        idx_partitions = []
                        for i in range(0, test_llrs.size, sizePartitions):
                            idx_partitions.append(list(idx_numbers[i:i+sizePartitions]))
                        for i in range(K):

                            idx_test = idx_partitions[i]
                            idx_train = idx_partitions[0:i] + idx_partitions[i+1:]

                            # from lists of lists collapse the elemnts in a single list
                            idx_train = sum(idx_train, [])

                            # partition the data and labels using the already partitioned indexes
                            STR = test_llrs[idx_train]
                            STE = test_llrs[idx_test]
                            LTRS = LTE[idx_train]
                            LTES = LTE[idx_test]
                            
                            cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(STR,LTRS,STE,l,0.5)
                            llr_cal.append(cal_llrs)
                            labels_cal.append(LTES)
                        llr_cal = numpy.hstack(llr_cal)
                        labels_cal = numpy.hstack(labels_cal)
                        DCF_act = BayesDecision.compute_act_DCF(llr_cal, labels_cal, pi1, Cfn, Cfp)
                        print("DCF calibrated act = ", DCF_act)
                        
                if BALANCING:
                    print("SVM Linear with balancing: K = %f, C = %f" % (K_,C), "\n")
                    test_llrs = LinearSVM.train_SVM_linear(DTR, LTR, DTE, C, K_, True, 0.5)   
                    DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)

            if POL_SVM:        
                K_list = [1]
                C_list = [1e-5,1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100] 
                c_list = [0,1,10,30]
                for K_ in K_list:
                    for c in c_list:  
                        for C in C_list:
                            
                            print("SVM Polynomial Kernel: K = %f, C = %f, d=2, c= %f" % (K_,C,c), "\n")
                            test_llrs = KernelSVM.kernel_svm(DTR, LTR, DTE, C, c, None, K_, "poly")  
                            DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                            DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                            print("DCF min= ", DCF_min)
                            print("DCF act = ", DCF_act)
                            
                if CALIBRATION:
                    for l in lambda_list:
                        print(" calibration with logistic regression with lamb ", l)
                        K=3
                        sizePartitions=int(test_llrs.size/K)
                        llr_cal = []
                        labels_cal = []
                        idx_numbers = numpy.arange(test_llrs.size)
                        idx_partitions = []
                        for i in range(0, test_llrs.size, sizePartitions):
                            idx_partitions.append(list(idx_numbers[i:i+sizePartitions]))
                        for i in range(K):

                            idx_test = idx_partitions[i]
                            idx_train = idx_partitions[0:i] + idx_partitions[i+1:]

                            # from lists of lists collapse the elemnts in a single list
                            idx_train = sum(idx_train, [])

                            # partition the data and labels using the already partitioned indexes
                            STR = test_llrs[idx_train]
                            STE = test_llrs[idx_test]
                            LTRS = LTE[idx_train]
                            LTES = LTE[idx_test]
                            
                            cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(STR,LTRS,STE,l,0.5)
                            llr_cal.append(cal_llrs)
                            labels_cal.append(LTES)
                        llr_cal = numpy.hstack(llr_cal)
                        labels_cal = numpy.hstack(labels_cal)
                        DCF_act = BayesDecision.compute_act_DCF(llr_cal, labels_cal, pi1, Cfn, Cfp)
                        print("DCF calibrated act = ", DCF_act)
                            
                if BALANCING:
                    print("SVM Polynomial Kernel with balancing: K = %f, C = %f, d=2, c= %f" % (K_,C,c), "\n")
                    test_llrs = KernelSVM.kernel_svm(DTR, LTR, DTE, C, c, None, K_, "poly", True, 0.5) 
                    DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)
                    listMinDCF.append(DCF_min)   
                    
            if RBF_SVM:       
                K_list = [1]
                C_list = [1]   
                C_list = [1e-5,1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]              
                g_list = [1e-5,1e-4,1e-3,1e-2]
                for K_ in K_list:
                    for g in g_list:  
                        for C in C_list:   
                            
                            print("SVM RBF Kernel: K = %f, C = %f, g=%f" % (K_,C,g), "\n")
                            test_llrs = KernelSVM.kernel_svm(DTR, LTR, DTE, C, None, g, K_, "RBF")  
                            DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                            DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                            print("DCF min= ", DCF_min)
                            print("DCF act = ", DCF_act)
                            #listMinDCF.append(DCF_min)
                            
                if CALIBRATION:
                    for l in lambda_list:
                        print(" calibration with logistic regression with lamb ", l)
                        K=3
                        sizePartitions=int(test_llrs.size/K)
                        llr_cal = []
                        labels_cal = []
                        idx_numbers = numpy.arange(test_llrs.size)
                        idx_partitions = []
                        for i in range(0, test_llrs.size, sizePartitions):
                            idx_partitions.append(list(idx_numbers[i:i+sizePartitions]))
                        for i in range(K):

                            idx_test = idx_partitions[i]
                            idx_train = idx_partitions[0:i] + idx_partitions[i+1:]

                            # from lists of lists collapse the elemnts in a single list
                            idx_train = sum(idx_train, [])

                            # partition the data and labels using the already partitioned indexes
                            STR = test_llrs[idx_train]
                            STE = test_llrs[idx_test]
                            LTRS = LTE[idx_train]
                            LTES = LTE[idx_test]
                            
                            cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(STR,LTRS,STE,l,0.5)
                            llr_cal.append(cal_llrs)
                            labels_cal.append(LTES)
                        llr_cal = numpy.hstack(llr_cal)
                        labels_cal = numpy.hstack(labels_cal)
                        DCF_act = BayesDecision.compute_act_DCF(llr_cal, labels_cal, pi1, Cfn, Cfp)
                        print("DCF calibrated act = ", DCF_act)
                            
                if BALANCING:
                    print("SVM RBF Kernel with balancing: K = %f, C = %f, g=%f" % (K_,C,g), "\n")
                    test_llrs = KernelSVM.kernel_svm(DTR, LTR, DTE, C, None, g, K_, "RBF", True, 0.5)
                    DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)
                    listMinDCF.append(DCF_min)
                    
            if FULL_GMM:
                psi = 0.01
                M_list = [2,4,8] 
                for M in M_list:
                    
                    print("GMM version = %s, M = %d, psi = %f" % ("full", M, psi), "\n")
                    test_llrs = gmm.GMM_classifier(DTR, LTR, DTE, M, psi, "full") 
                    DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)  
                    
                if CALIBRATION:
                    for l in lambda_list:
                        print(" calibration with logistic regression with lamb ", l)
                        K=3
                        sizePartitions=int(test_llrs.size/K)
                        llr_cal = []
                        labels_cal = []
                        idx_numbers = numpy.arange(test_llrs.size)
                        idx_partitions = []
                        for i in range(0, test_llrs.size, sizePartitions):
                            idx_partitions.append(list(idx_numbers[i:i+sizePartitions]))
                        for i in range(K):

                            idx_test = idx_partitions[i]
                            idx_train = idx_partitions[0:i] + idx_partitions[i+1:]

                            # from lists of lists collapse the elemnts in a single list
                            idx_train = sum(idx_train, [])

                            # partition the data and labels using the already partitioned indexes
                            STR = test_llrs[idx_train]
                            STE = test_llrs[idx_test]
                            LTRS = LTE[idx_train]
                            LTES = LTE[idx_test]
                            
                            cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(STR,LTRS,STE,l,0.5)
                            llr_cal.append(cal_llrs)
                            labels_cal.append(LTES)
                        llr_cal = numpy.hstack(llr_cal)
                        labels_cal = numpy.hstack(labels_cal)
                        DCF_act = BayesDecision.compute_act_DCF(llr_cal, labels_cal, pi1, Cfn, Cfp)
                        print("DCF calibrated act = ", DCF_act)
                    

            if DIAG_GMM:
                psi = 0.01
                M_list = [2,4,8] 
                for M in M_list:
                    
                    print("GMM version = %s, M = %d, psi = %f" % ("diagonal", M, psi), "\n")
                    test_llrs = gmm.GMM_classifier(DTR, LTR, DTE, M, psi, "diagonal")  
                    DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)         
                    
                if CALIBRATION:
                    for l in lambda_list:
                        print(" calibration with logistic regression with lamb ", l)
                        K=3
                        sizePartitions=int(test_llrs.size/K)
                        llr_cal = []
                        labels_cal = []
                        idx_numbers = numpy.arange(test_llrs.size)
                        idx_partitions = []
                        for i in range(0, test_llrs.size, sizePartitions):
                            idx_partitions.append(list(idx_numbers[i:i+sizePartitions]))
                        for i in range(K):

                            idx_test = idx_partitions[i]
                            idx_train = idx_partitions[0:i] + idx_partitions[i+1:]

                            # from lists of lists collapse the elemnts in a single list
                            idx_train = sum(idx_train, [])

                            # partition the data and labels using the already partitioned indexes
                            STR = test_llrs[idx_train]
                            STE = test_llrs[idx_test]
                            LTRS = LTE[idx_train]
                            LTES = LTE[idx_test]
                            
                            cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(STR,LTRS,STE,l,0.5)
                            llr_cal.append(cal_llrs)
                            labels_cal.append(LTES)
                        llr_cal = numpy.hstack(llr_cal)
                        labels_cal = numpy.hstack(labels_cal)
                        DCF_act = BayesDecision.compute_act_DCF(llr_cal, labels_cal, pi1, Cfn, Cfp)
                        print("DCF calibrated act = ", DCF_act) 
            
            if TIED_GMM:
                psi = 0.01
                M_list = [2,4,8] 
                for M in M_list:
                    
                    print("GMM version = %s, M = %d, psi = %f" % ("tied", M, psi), "\n")
                    test_llrs = gmm.GMM_classifier(DTR, LTR, DTE, M, psi, "tied")  
                    DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)
                    
                if CALIBRATION:
                    for l in lambda_list:
                        print(" calibration with logistic regression with lamb ", l)
                        K=3
                        sizePartitions=int(test_llrs.size/K)
                        llr_cal = []
                        labels_cal = []
                        idx_numbers = numpy.arange(test_llrs.size)
                        idx_partitions = []
                        for i in range(0, test_llrs.size, sizePartitions):
                            idx_partitions.append(list(idx_numbers[i:i+sizePartitions]))
                        for i in range(K):

                            idx_test = idx_partitions[i]
                            idx_train = idx_partitions[0:i] + idx_partitions[i+1:]

                            # from lists of lists collapse the elemnts in a single list
                            idx_train = sum(idx_train, [])

                            # partition the data and labels using the already partitioned indexes
                            STR = test_llrs[idx_train]
                            STE = test_llrs[idx_test]
                            LTRS = LTE[idx_train]
                            LTES = LTE[idx_test]
                            
                            cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(STR,LTRS,STE,l,0.5)
                            llr_cal.append(cal_llrs)
                            labels_cal.append(LTES)
                        llr_cal = numpy.hstack(llr_cal)
                        labels_cal = numpy.hstack(labels_cal)
                        DCF_act = BayesDecision.compute_act_DCF(llr_cal, labels_cal, pi1, Cfn, Cfp)
                        print("DCF calibrated act = ", DCF_act)

            """Fusion of our best models: Tied GMM with M=8 and SVM RBF with C=1 g=1e-3"""      
            
            if FUSION:
                K=3
                #Tied GMM m=8 0.031
                #RBF c=1 g=1e-3 0.039
                
                #First model
                psi = 0.01
                M=8
                print("GMM version = %s, M = %d, psi = %f" % ("tied", M, psi), "\n")
                all_llrs1 = gmm.GMM_classifier(DTR, LTR, DTE, M, psi, "tied")   
                DCF_min =  BayesDecision.compute_min_DCF(all_llrs1, LTE, pi1, Cfn, Cfp)
                DCF_act = BayesDecision.compute_act_DCF(all_llrs1, LTE, pi1, Cfn, Cfp)
                print("DCF min= ", DCF_min)
                print("DCF act = ", DCF_act)
                    
                #Second model  
                K_ = 1
                C = 1                 
                g = 1e-3  
                print("SVM RBF Kernel: K = %f, C = %f, g=%f" % (K_,C,g), "\n")
                all_llrs2 = KernelSVM.kernel_svm(DTR, LTR, DTE, C, None, g, K_, "RBF")
                DCF_min =  BayesDecision.compute_min_DCF(all_llrs2, LTE, pi1, Cfn, Cfp)
                DCF_act = BayesDecision.compute_act_DCF(all_llrs2, LTE, pi1, Cfn, Cfp)
                print("DCF min= ", DCF_min)
                print("DCF act = ", DCF_act)
                
                
                test_llrs_stacked=numpy.vstack((all_llrs1,all_llrs2))
                print(test_llrs_stacked.shape)

                llr_fus = []
                labels_fus = []
                idx_numbers = numpy.arange(test_llrs_stacked.shape[1])
                idx_partitions = []
                sizePartitions = int(test_llrs_stacked.shape[1]/K)
                for i in range(0, test_llrs_stacked.shape[1], sizePartitions):
                    idx_partitions.append(list(idx_numbers[i:i+sizePartitions]))
                for i in range(K):

                    idx_test = idx_partitions[i]
                    idx_train = idx_partitions[0:i] + idx_partitions[i+1:]

                    # from lists of lists collapse the elemnts in a single list
                    idx_train = sum(idx_train, [])

                    # partition the data and labels using the already partitioned indexes
                    STR = test_llrs_stacked[:,idx_train]
                    STE = test_llrs_stacked[:,idx_test]
                    LTRS = LTE[idx_train]
                    LTES = LTE[idx_test]
                    
                    print("fusion")
                    fus_llr= LinearLogisticRegression.LinearLogisticRegression(STR,LTRS, STE, 1e-03, True, 0.5)
                    print(fus_llr.shape)
                    llr_fus.append(fus_llr)
                    labels_fus.append(LTES)

                llr_fus = numpy.hstack(llr_fus)
                labels_fus = numpy.hstack(labels_fus)

                DCF_min =  BayesDecision.compute_min_DCF(llr_fus, labels_fus, pi1, Cfn, Cfp)
                DCF_act = BayesDecision.compute_act_DCF(llr_fus, labels_fus, pi1, Cfn, Cfp)
                print("DCF min= ", DCF_min)
                print("DCF act = ", DCF_act)
        