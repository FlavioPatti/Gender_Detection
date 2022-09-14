import numpy
import matplotlib.pyplot as plt

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

#at the end we have reset all the options to 0
FLAG_SHOW_FIGURES_INIT = 1
FLAG_SHOW_FIGURES_END = 0

FLAG_TRAINING= 1
FLAG_TESTING= 0

FLAG_CALIBRATION=0
FLAG_BALANCING=0
FLAG_FUSION=0

FLAG_SINGLEFOLD= 1

FLAG_GAUSSIANIZATION= 0
FLAG_ZNORMALIZATION=0
FLAG_PCA = 0
FLAG_LDA = 0

FLAG_MVG = 0
FLAG_NAIVE =0
FLAG_TIED = 0
FLAG_LOGISTIC = 0
FLAG_SVM= 0
FLAG_GMM= 0

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
        sample = vcol(numpy.array(line[0:12], dtype=numpy.float32))
        DTR.append(sample)
        LTR.append(line[-1])
    f.close()
    DTE = []
    LTE = []   
    f=open('data/Test.txt', encoding="ISO-8859-1")
    for line in f:
        line = line.strip().split(',')
        sample = vcol(numpy.array(line[0:12], dtype=numpy.float32))
        DTE.append(sample)
        LTE.append(line[-1])
    f.close()
    return numpy.hstack(DTR), numpy.array(LTR, dtype=numpy.int32), numpy.hstack(DTE), numpy.array(LTE, dtype=numpy.int32)  

def split_db_2to1(D, L, param, seed=0):
    """ Split the dataset in two parts based on the param,
        first part will be used for model training, second part for validation
        D is the dataset, L the corresponding labels
        seed is set to 0 and it's used to randomize partitions
        returns:
        DTR_TRA = Dataset for training set
        LTR_TRA = Labels for training set
        DTR_TEST = Dataset for validation set
        LTR_TEST = Labels for validation set
    """
    nTrain = int(D.shape[1]*param)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxVal = idx[nTrain:]
    DTR_TRA = D[:, idxTrain]
    DTR_TEST = D[:, idxVal]
    LTR_TRA = L[idxTrain]
    LTR_TEST = L[idxVal]
    return (DTR_TRA, LTR_TRA), (DTR_TEST, LTR_TEST)


if __name__ == '__main__':

    #load data
    DTR, LTR, DTE, LTE = load_train_and_test()
    print("Total sample of training set: ", DTR.shape[1] )
    print("Total sample of test set: ", DTE.shape[1] )
    
    print("Total sample of class 0 for training set: ", (LTR==0).sum())
    print("Total sample of class 1 for training set: ", (LTR==1).sum())
    
    print("Total sample of class 0 for test set: ", (LTE==0).sum())
    print("Total sample of class 1 for test set: ", (LTE==1).sum())
    
    hFea = {
        0: 'Feature 0',
        1: 'Feature 1',
        2: 'Feature 2',
        3: 'Feature 3',
        4: 'Feature 4',
        5: 'Feature 5',
        6: 'Feature 6',
        7: 'Feature 7',
        8: 'Feature 8',
        9: 'Feature 9',
        10: 'Feature 10',
        11: 'Feature 11'
        }
    
    DTR0=DTR[:,LTR==0]
    DTR1=DTR[:,LTR==1]
    
    # 3 applications: main balanced one and two unbalanced 
    applications = [[0.5,1,1],[0.1,1,1],[0.9,1,1]]

    """Flag useful to generate graphics of the various attributes of our data set"""
    if FLAG_SHOW_FIGURES_INIT:
        
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

    if FLAG_TRAINING:
        print("training")
        lambda_list = [0, 1e-6, 1e-3,1]
        if FLAG_SHOW_FIGURES_END:
            listMinDCF=[]
        
        if FLAG_SINGLEFOLD:
            print("Singlefold")
            
            """We decide to split our data in 50% training, 30% validation and 20% testing"""
            (DTR_TRA, LTR_TRA), (DTR_TEST, LTR_TEST)=split_db_2to1(DTR,LTR,0.8)
            (DTR_TRA, LTR_TRA), (DTR_V,LTR_V) = split_db_2to1(DTR_TRA, LTR_TRA,0.625)
            sample_class0 = (LTR_TRA==0).sum()
            print("Sample of class 0: ", sample_class0)
            sample_class1 = (LTR_TRA==1).sum()
            print("Sample of class 1: ", sample_class1, "\n")

            if FLAG_GAUSSIANIZATION:
                DTR_TEST = Gaussianization.Gaussianization(DTR_TRA,DTR_TEST)
                DTR_V = Gaussianization.Gaussianization(DTR_TRA,DTR_V)
                DTR_TRA = Gaussianization.Gaussianization(DTR_TRA,DTR_TRA)
                print("Gaussianization")

            if FLAG_ZNORMALIZATION:
                DTR_TEST = ZNormalization.ZNormalization(DTR_TRA,DTR_TEST)
                DTR_V = ZNormalization.ZNormalization(DTR_TRA,DTR_V)
                DTR_TRA = ZNormalization.ZNormalization(DTR_TRA,DTR_TRA)
                print("Z-normalization")   
                 
            if FLAG_PCA:
                DTR_TEST=pca.PCA(DTR_TRA, DTR_TEST, 7)
                DTR_V=pca.PCA(DTR_TRA, DTR_V, 7)
                DTR_TRA=pca.PCA(DTR_TRA, DTR_TRA, 7)
                print("PCA dimensionality: ",DTR_TRA.shape)

            if FLAG_LDA:
                DTR_TEST=lda.LDA(DTR_TRA, LTR_TRA,DTR_TEST, 1)
                DTR_V=lda.LDA(DTR_TRA, LTR_TRA,DTR_V, 1)
                DTR_TRA=lda.LDA(DTR_TRA, LTR_TRA,DTR_TRA, 1)
                print("LDA dimensionality: ",DTR_TRA.shape)

            for app in applications:
                pi1, Cfn, Cfp = app
                print("Application: pi1 = %.1f, Cfn = %d, Cfp = %d" %(pi1, Cfn,Cfp))
            
                if FLAG_MVG:
                    print("mvg")
                    test_llrs = MultivariateGaussianClassifier.MultivariateGaussianClassifier(DTR_TRA, LTR_TRA, DTR_TEST)
                    DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)
                    
                    if FLAG_CALIBRATION:
                        for l in lambda_list:
                            print(" calibration with logistic regression with lamb ", l)
                            tra_llrs = MultivariateGaussianClassifier.MultivariateGaussianClassifier(DTR_TRA, LTR_TRA, DTR_V)
                            cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(tra_llrs,LTR_V,test_llrs,l,0.5)
                            DCF_act = BayesDecision.compute_act_DCF(cal_llrs, LTR_TEST, pi1, Cfn, Cfp)
                            print("DCF calibrated act = ", DCF_act)
                    
                if FLAG_NAIVE:
                    print("naive")
                    test_llrs = NaiveBayesClassifier.NaiveBayesClassifier(DTR_TRA, LTR_TRA, DTR_TEST)
                    DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)
                    
                    if FLAG_CALIBRATION:
                        for l in lambda_list:
                            print(" calibration with logistic regression with lamb ", l)
                            tra_llrs = NaiveBayesClassifier.NaiveBayesClassifier(DTR_TRA, LTR_TRA, DTR_V)
                            cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(tra_llrs,LTR_V,test_llrs,l,0.5)
                            DCF_act = BayesDecision.compute_act_DCF(cal_llrs, LTR_TEST, pi1, Cfn, Cfp)
                            print("DCF calibrated act = ", DCF_act)

                if FLAG_TIED:
                    print("tied gaussian")
                    test_llrs = TiedGaussianClassifier.TiedGaussianClassifier(DTR_TRA, LTR_TRA, DTR_TEST)
                    DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)
                    
                    if FLAG_CALIBRATION:
                        for l in lambda_list:
                            print(" calibration with logistic regression with lamb ", l)
                            tra_llrs = TiedGaussianClassifier.TiedGaussianClassifier(DTR_TRA, LTR_TRA, DTR_V)
                            cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(tra_llrs,LTR_V,test_llrs,l,0.5)
                            DCF_act = BayesDecision.compute_act_DCF(cal_llrs, LTR_TEST, pi1, Cfn, Cfp)
                            print("DCF calibrated act = ", DCF_act)
                            
                    print("tied naive")
                    test_llrs = TiedNaiveBayes.TiedNaiveBayes(DTR_TRA, LTR_TRA, DTR_TEST)
                    DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)
                    
                    if FLAG_CALIBRATION:
                        for l in lambda_list:
                            print(" calibration with logistic regression with lamb ", l)
                            tra_llrs = TiedNaiveBayes.TiedNaiveBayes(DTR_TRA, LTR_TRA, DTR_V)
                            cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(tra_llrs,LTR_V,test_llrs,l,0.5)
                            DCF_act = BayesDecision.compute_act_DCF(cal_llrs, LTR_TEST, pi1, Cfn, Cfp)
                            print("DCF calibrated act = ", DCF_act)

                if FLAG_LOGISTIC:
                    for l in lambda_list:
                    
                        print(" linear logistic regression with lamb ", l)
                        test_llrs = LinearLogisticRegression.LinearLogisticRegression(DTR_TRA, LTR_TRA, DTR_TEST, l)
                        DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                        DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                        print("DCF min= ", DCF_min)
                        print("DCF act = ", DCF_act)
                       # listMinDCF.append(DCF_min)
                        
                        if FLAG_CALIBRATION:
                            for l2 in lambda_list:
                                print(" calibration with logistic regression with lamb ", l2)
                                tra_llrs = LinearLogisticRegression.LinearLogisticRegression(DTR_TRA, LTR_TRA, DTR_V,l)
                                cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(tra_llrs,LTR_V,test_llrs,l2,0.5)
                                DCF_act = BayesDecision.compute_act_DCF(cal_llrs, LTR_TEST, pi1, Cfn, Cfp)
                                print("DCF calibrated act = ", DCF_act)
                        
                        if FLAG_BALANCING:
                            print(" balancing of the linear logistic regression with lamb ", l)
                            test_llrs = LinearLogisticRegression.BalancedLinearLogisticRegression(DTR_TRA, LTR_TRA, DTR_TEST, l,0.5)
                            DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                            DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                            print("DCF min= ", DCF_min)
                            print("DCF act = ", DCF_act)
                            #listMinDCF.append(DCF_min)
                            
                        
                        print(" quadratic logistic regression with lamb ", l)
                        test_llrs = QuadraticLogisticRegression.QuadraticLogisticRegression(DTR_TRA, LTR_TRA, DTR_TEST, l)
                        DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                        DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                        print("DCF min= ", DCF_min)
                        print("DCF act = ", DCF_act)
                       # listMinDCF.append(DCF_min)
                        
                        """
                       #plot bayes error
                        p = numpy.linspace(-3,3,21)
                        plt.plot(p, graphics.bayes_error_plot(p, test_llrs, LTR_TEST,minCost=False), color='r')
                        plt.plot(p, graphics.bayes_error_plot(p, test_llrs, LTR_TEST,minCost=True), color='b')
                        plt.ylim([0, 1.1])
                        plt.xlim([-3, 3])
                        plt.savefig('Graphics/Generated_figures/DCFPlots/Quadratic LR-minDCF-actDCF.jpg')
                        plt.show()
                        """
                        
                        if FLAG_CALIBRATION:
                            for l2 in lambda_list:
                                print(" calibration with logistic regression with lamb ", l2)
                                tra_llrs = QuadraticLogisticRegression.QuadraticLogisticRegression(DTR_TRA, LTR_TRA, DTR_V, l)
                                cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(tra_llrs,LTR_V,test_llrs,l2,0.5)
                                DCF_act = BayesDecision.compute_act_DCF(cal_llrs, LTR_TEST, pi1, Cfn, Cfp)
                                print("DCF calibrated act = ", DCF_act)
                                
                                """
                                #plot bayes error
                                p = numpy.linspace(-3,3,21)
                                plt.plot(p, graphics.bayes_error_plot(p, cal_llrs, LTR_TEST,minCost=False), color='r')
                                plt.plot(p, graphics.bayes_error_plot(p, cal_llrs, LTR_TEST,minCost=True), color='b')
                                plt.ylim([0, 1.1])
                                plt.xlim([-3, 3])
                                plt.savefig('Graphics/Generated_figures/DCFPlots/Calibrated Quadratic LR with lamb=%f-minDCF-actDCF.jpg' % l2)
                                plt.show()
                                """
                                
                        if FLAG_BALANCING:
                            print(" balancing of the quadratic logistic regression with lamb ", l)
                            test_llrs = QuadraticLogisticRegression.BalancedQuadraticLogisticRegression(DTR_TRA, LTR_TRA, DTR_TEST, l,0.5)
                            DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                            DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                            print("DCF min= ", DCF_min)
                            print("DCF act = ", DCF_act)
                           # listMinDCF.append(DCF_min)
                        
                                
                if FLAG_SVM:
                    
                    K_list = [1, 10]
                    C_list = [1e-3, 0.1, 1]
                    for K_ in K_list:
                        for C in C_list:
                            
                            print("SVM Linear: K = %f, C = %f" % (K_,C), "\n")
                            test_llrs = LinearSVM.train_SVM_linear(DTR_TRA, LTR_TRA, DTR_TEST, C, K_, 0, balanced = False)  
                            DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                            DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                            print("DCF min= ", DCF_min)
                            print("DCF act = ", DCF_act)
                            
                            if FLAG_BALANCING:
                                print("SVM Linear with balancing: K = %f, C = %f" % (K_,C), "\n")
                                test_llrs = LinearSVM.train_SVM_linear(DTR_TRA, LTR_TRA, DTR_TEST, C, K_, 0.5, balanced= True)  
                                DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                                DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                                print("DCF min= ", DCF_min)
                                print("DCF act = ", DCF_act)
                            
                            if FLAG_CALIBRATION:
                                for l in lambda_list:
                                    print(" calibration with logistic regression with lamb ", l)
                                    tra_llrs = LinearSVM.train_SVM_linear(DTR_TRA, LTR_TRA, DTR_V, C, K_, 0.5, balanced= True)
                                    cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(tra_llrs,LTR_V,test_llrs,l,0.5)
                                    DCF_act = BayesDecision.compute_act_DCF(cal_llrs, LTR_TEST, pi1, Cfn, Cfp)
                                    print("DCF calibrated act = ", DCF_act)
                        
                    K_list = [1]
                    C_list = [1e-5,1e-4,1e-3,1e-2,1e-1]       
                    c_list = [0,1,10,30]
                    for K_ in K_list:
                        for c in c_list:  
                            for C in C_list:
                                
                                print("SVM Polynomial Kernel: K = %f, C = %f, d=2, c= %f" % (K_,C,c), "\n")
                                test_llrs = KernelSVM.quad_kernel_svm(DTR_TRA, LTR_TRA, DTR_TEST, C, c, 0, K_, "poly", balanced = False)
                                DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                                DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                                print("DCF min= ", DCF_min)
                                print("DCF act = ", DCF_act)
                                
                                if FLAG_BALANCING:
                                    print("SVM Polynomial Kernel with balancing: K = %f, C = %f, d=2, c= %f" % (K_,C,c), "\n")
                                    test_llrs = KernelSVM.quad_kernel_svm(DTR_TRA, LTR_TRA, DTR_TEST, C, c, 0, K_, "poly", balanced = True)
                                    DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                                    DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                                    print("DCF min= ", DCF_min)
                                    print("DCF act = ", DCF_act)
                                    listMinDCF.append(DCF_min)   
                                
                                if FLAG_CALIBRATION:
                                    for l in lambda_list:
                                        print(" calibration with logistic regression with lamb ", l)
                                        tra_llrs = KernelSVM.quad_kernel_svm(DTR_TRA, LTR_TRA, DTR_V, C, c, 0, K_, "poly", balanced = True)
                                        cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(tra_llrs,LTR_V,test_llrs,l,0.5)
                                        DCF_act = BayesDecision.compute_act_DCF(cal_llrs, LTR_TEST, pi1, Cfn, Cfp)
                                        print("DCF calibrated act = ", DCF_act)
                     
                    
                           
                    K_list = [1]
                    C_list = [0.001,0.01,0.1,1,10,100,1000]                 
                    g_list = [1e-5,1e-4,1e-3,1e-2]
                    for K_ in K_list:
                        for g in g_list:  
                            for C in C_list:   
                                
                                print("SVM RBF Kernel: K = %f, C = %f, g=%f" % (K_,C,g), "\n")
                                test_llrs = KernelSVM.quad_kernel_svm(DTR_TRA, LTR_TRA, DTR_TEST, C, 0, g, K_, "RBF", balanced = False)
                                DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                                DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                                print("DCF min= ", DCF_min)
                                print("DCF act = ", DCF_act)
                                #listMinDCF.append(DCF_min)
                                
                                if FLAG_BALANCING:
                                    print("SVM RBF Kernel with balancing: K = %f, C = %f, g=%f" % (K_,C,g), "\n")
                                    test_llrs1 = KernelSVM.quad_kernel_svm(DTR_TRA, LTR_TRA, DTR_TEST, C, 0, g, K_, "RBF", balanced = True)
                                    DCF_min =  BayesDecision.compute_min_DCF(test_llrs1, LTR_TEST, pi1, Cfn, Cfp)
                                    DCF_act = BayesDecision.compute_act_DCF(test_llrs1, LTR_TEST, pi1, Cfn, Cfp)
                                    print("DCF min= ", DCF_min)
                                    print("DCF act = ", DCF_act)
                                    listMinDCF.append(DCF_min)
                                
                                if FLAG_CALIBRATION:
                                    for l in lambda_list:
                                        print(" calibration with logistic regression with lamb ", l)
                                        tra_llrs = KernelSVM.quad_kernel_svm(DTR_TRA, LTR_TRA, DTR_V, C, 0, g, K_, "RBF", balanced = True)
                                        cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(tra_llrs,LTR_V,test_llrs,l,0.5)
                                        DCF_act = BayesDecision.compute_act_DCF(cal_llrs, LTR_TEST, pi1, Cfn, Cfp)
                                        print("DCF calibrated act = ", DCF_act)
                    
                    
                if FLAG_GMM:
                    psi = 0.01
                    M_list = [2,4,8] 
                    versions = ["full","diagonal","tied"]
                    for version in versions:
                        for M in M_list:
                           
                            print("GMM version = %s, M = %d, psi = %f" % (version, M, psi), "\n")
                            test_llrs= gmm.GMM_classifier(DTR_TRA, LTR_TRA, DTR_TEST, M, psi, version) 
                            DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                            DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTR_TEST, pi1, Cfn, Cfp)
                            print("DCF min= ", DCF_min)
                            print("DCF act = ", DCF_act)
                            
                            if FLAG_CALIBRATION:
                                for l in lambda_list:
                                    print(" calibration with logistic regression with lamb ", l)
                                    tra_llrs = gmm.GMM_classifier(DTR_TRA, LTR_TRA, DTR_V, M, psi, version)
                                    cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(tra_llrs,LTR_V,test_llrs,l,0.5)
                                    DCF_act = BayesDecision.compute_act_DCF(cal_llrs, LTR_TEST, pi1, Cfn, Cfp)
                                    print("DCF calibrated act = ", DCF_act)  
                            
                
                     
                """Fusion of our best models: Balanced SVM Linear K=1, C=0.1 with PCA=7 and Z-Normalized Quadratic LR with lambda=1e-3"""     
                
                if FLAG_FUSION:
                    """First model"""
                    
                    DTR_TEST1=pca.PCA(DTR_TRA, DTR_TEST, 7)
                    DTR_V1=pca.PCA(DTR_TRA, DTR_V, 7)
                    DTR_TRA1=pca.PCA(DTR_TRA, DTR_TRA, 7)
                    print("PCA dimensionality: ",DTR_TRA.shape)
                    
                    print("SVM Linear with balancing: K = %f, C = %f" % (1,0.1), "\n")
                    test_llrs1 = LinearSVM.train_SVM_linear(DTR_TRA1, LTR_TRA, DTR_TEST1, 0.1, 1, 0.5, balanced= True)  
                    DCF_min =  BayesDecision.compute_min_DCF(test_llrs1, LTR_TEST, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(test_llrs1, LTR_TEST, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)
                        
                    """Second model"""
                    
                    DTR_TEST2 = ZNormalization.ZNormalization(DTR_TRA,DTR_TEST)
                    DTR_V2 = ZNormalization.ZNormalization(DTR_TRA,DTR_V)
                    DTR_TRA2 = ZNormalization.ZNormalization(DTR_TRA,DTR_TRA)
                    print("Z-normalization")  
                    
                    print(" quadratic logistic regression with lamb ", 1e-3)
                    test_llrs2 = QuadraticLogisticRegression.QuadraticLogisticRegression(DTR_TRA2, LTR_TRA, DTR_TEST2, 1e-3)
                    DCF_min =  BayesDecision.compute_min_DCF(test_llrs2, LTR_TEST, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(test_llrs2, LTR_TEST, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)
                    
                    
                    test_llrs_stacked=numpy.vstack((test_llrs1,test_llrs2))
                    print(test_llrs_stacked.shape)
                    
                    tra_llrs1 = LinearSVM.train_SVM_linear(DTR_TRA1, LTR_TRA, DTR_V1, 0.1, 1, 0.5, balanced= True) 
                    tra_llrs2 = QuadraticLogisticRegression.QuadraticLogisticRegression(DTR_TRA2, LTR_TRA, DTR_V2, 1e-3) 
               
                    tra_llrs_stacked=numpy.vstack((tra_llrs1,tra_llrs2))
                    print(tra_llrs_stacked.shape)
                    print("fusion")
                    fus_llrs= LinearLogisticRegression.BalancedLinearLogisticRegression(tra_llrs_stacked,LTR_V,test_llrs_stacked, 1e-03,0.5)
                    print(fus_llrs.shape)
                    DCF_min =  BayesDecision.compute_min_DCF(fus_llrs, LTR_TEST, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(fus_llrs, LTR_TEST, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)
                      
              
            """plot of the minDCF at the end of the computation"""
            """
            if FLAG_SHOW_FIGURES_END:
                lambda_list_plot = [1e-12, 1e-6, 1e-3,1]
                print("listMinDCF lenght: ", len(listMinDCF))
                graphics.plotDCFprior(lambda_list_plot,listMinDCF,"lambda Balanced Linear LR")
            """
            """
            if FLAG_SHOW_FIGURES_END:
                C_list_plot=[1e-5,1e-4,1e-3,1e-2,1e-1]
                print("listMinDCF lenght: ", len(listMinDCF))
                graphics.plotDCFc(C_list_plot,listMinDCF,"C Balanced SVM polynomial kernel")
            """
            """
            if FLAG_SHOW_FIGURES_END:
                C_list_plot=[0.001,0.01,0.1,1,10,100,1000]
                print("listMinDCF lenght: ", len(listMinDCF))
                graphics.plotDCFg(C_list_plot,listMinDCF,"C Balanced SVM RBF kernel")
            """
            
    if FLAG_TESTING:
        print("testing")
        lambda_list = [0, 1e-6, 1e-3, 1]
        listMinDCF=[]
        
        """ we performed testing using 1/2 of the original training set for training, and the evaluation/test set for evaluating our choices"""
        if FLAG_SINGLEFOLD:
            print("Singlefold")
            (DTR_TRA, LTR_TRA), (DTR_V,LTR_V) = split_db_2to1(DTR, LTR,0.5)
            sample_class0 = (LTR==0).sum()
            print("Sample of class 0: ", sample_class0)
            sample_class1 = (LTR==1).sum()
            print("Sample of class 1: ", sample_class1, "\n")
            if FLAG_GAUSSIANIZATION:
                DTE = Gaussianization.Gaussianization(DTR_TRA,DTE)
                DTR_V = Gaussianization.Gaussianization(DTR_TRA,DTR_V)
                DTR_TRA = Gaussianization.Gaussianization(DTR_TRA,DTR_TRA)
                print("Gaussianization")

            if FLAG_ZNORMALIZATION:
                DTE = ZNormalization.ZNormalization(DTR_TRA,DTE)
                DTR_V = ZNormalization.ZNormalization(DTR_TRA,DTR_V)
                DTR_TRA = ZNormalization.ZNormalization(DTR_TRA,DTR_TRA)
                print("Z-normalization")   
                 
            if FLAG_PCA:
                DTE=pca.PCA(DTR_TRA, DTE, 7)
                DTR_V=pca.PCA(DTR_TRA, DTR_V, 7)
                DTR_TRA=pca.PCA(DTR_TRA, DTR_TRA, 7)
                print("PCA dimensionality: ",DTR_TRA.shape)

            if FLAG_LDA:
                DTE=lda.LDA(DTR_TRA, LTR_TRA,DTE, 1)
                DTR_V=lda.LDA(DTR_TRA, LTR_TRA,DTR_V, 1)
                DTR_TRA=lda.LDA(DTR_TRA, LTR_TRA,DTR_TRA, 1)
                print("LDA dimensionality: ",DTR_TRA.shape)

            for app in applications:
                pi1, Cfn, Cfp = app
                print("Application: pi1 = %.1f, Cfn = %d, Cfp = %d" %(pi1, Cfn,Cfp))
                
                """testing our best models with other kind of data"""
                
                if FLAG_LOGISTIC:
                    for l in lambda_list:        
                        print(" quadratic logistic regression with lamb ", l)
                        test_llrs = QuadraticLogisticRegression.QuadraticLogisticRegression(DTR_TRA, LTR_TRA, DTE, l)
                        DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                        DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                        print("DCF min= ", DCF_min)
                        print("DCF act = ", DCF_act)
                        # listMinDCF.append(DCF_min)
                    
                        if FLAG_CALIBRATION:
                            for l2 in lambda_list:
                                print(" calibration with logistic regression with lamb ", l2)
                                tra_llrs = QuadraticLogisticRegression.QuadraticLogisticRegression(DTR_TRA, LTR_TRA, DTR_V, l)
                                cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(tra_llrs,LTR_V,test_llrs,l2,0.5)
                                DCF_act = BayesDecision.compute_act_DCF(cal_llrs, LTE, pi1, Cfn, Cfp)
                                print("DCF calibrated act = ", DCF_act)
                        
                                
                if FLAG_SVM:
                    K_list = [1,10]
                    C_list = [1e-3,0.1, 1]
                    for K_ in K_list:
                        for C in C_list:  
                            if FLAG_BALANCING:
                                print("SVM Linear with balancing: K = %f, C = %f" % (K_,C), "\n")
                                test_llrs = LinearSVM.train_SVM_linear(DTR_TRA, LTR_TRA, DTE, C, K_, 0.5, balanced= True)  
                                DCF_min =  BayesDecision.compute_min_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                                DCF_act = BayesDecision.compute_act_DCF(test_llrs, LTE, pi1, Cfn, Cfp)
                                print("DCF min= ", DCF_min)
                                print("DCF act = ", DCF_act)
                                
                            if FLAG_CALIBRATION:
                                for l in lambda_list:
                                    print(" calibration with logistic regression with lamb ", l)
                                    tra_llrs = LinearSVM.train_SVM_linear(DTR_TRA, LTR_TRA, DTR_V, C, K_, 0.5, balanced= True)
                                    cal_llrs=LinearLogisticRegression.PriWeiLinearLogisticRegression(tra_llrs,LTR_V,test_llrs,l,0.5)
                                    DCF_act = BayesDecision.compute_act_DCF(cal_llrs, LTE, pi1, Cfn, Cfp)
                                    print("DCF calibrated act = ", DCF_act)                 
                                   
                """Fusion of the SVM RBF and GMM"""     
                
                if FLAG_FUSION:
                    """First model"""
                    
                    DTE1=pca.PCA(DTR_TRA, DTE, 7)
                    DTR_V1=pca.PCA(DTR_TRA, DTR_V, 7)
                    DTR_TRA1=pca.PCA(DTR_TRA, DTR_TRA, 7)
                    print("PCA dimensionality: ",DTR_TRA.shape)
                    
                    print("SVM Linear with balancing: K = %f, C = %f" % (1,0.1), "\n")
                    test_llrs1 = LinearSVM.train_SVM_linear(DTR_TRA1, LTR_TRA, DTE1, 0.1, 1, 0.5, balanced= True)  
                    DCF_min =  BayesDecision.compute_min_DCF(test_llrs1, LTE, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(test_llrs1, LTE, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)
                        
                    """Second model"""
                    
                    DTE2 = ZNormalization.ZNormalization(DTR_TRA,DTE)
                    DTR_V2 = ZNormalization.ZNormalization(DTR_TRA,DTR_V)
                    DTR_TRA2 = ZNormalization.ZNormalization(DTR_TRA,DTR_TRA)
                    print("Z-normalization")  
                    
                    print(" quadratic logistic regression with lamb ", 1e-3)
                    test_llrs2 = QuadraticLogisticRegression.QuadraticLogisticRegression(DTR_TRA2, LTR_TRA, DTE2, 1e-3)
                    DCF_min =  BayesDecision.compute_min_DCF(test_llrs2, LTE, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(test_llrs2, LTE, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)
                    
                    
                    test_llrs_stacked=numpy.vstack((test_llrs1,test_llrs2))
                    print(test_llrs_stacked.shape)
                    
                    tra_llrs1 = LinearSVM.train_SVM_linear(DTR_TRA1, LTR_TRA, DTR_V1, 0.1, 1, 0.5, balanced= True) 
                    tra_llrs2 = QuadraticLogisticRegression.QuadraticLogisticRegression(DTR_TRA2, LTR_TRA, DTR_V2, 1e-3) 
               
                    tra_llrs_stacked=numpy.vstack((tra_llrs1,tra_llrs2))
                    print(tra_llrs_stacked.shape)
                    print("fusion")
                    fus_llrs= LinearLogisticRegression.BalancedLinearLogisticRegression(tra_llrs_stacked,LTR_V,test_llrs_stacked, 1e-03,0.5)
                    print(fus_llrs.shape)
                    DCF_min =  BayesDecision.compute_min_DCF(fus_llrs, LTE, pi1, Cfn, Cfp)
                    DCF_act = BayesDecision.compute_act_DCF(fus_llrs, LTE, pi1, Cfn, Cfp)
                    print("DCF min= ", DCF_min)
                    print("DCF act = ", DCF_act)

