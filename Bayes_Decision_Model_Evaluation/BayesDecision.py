from ast import Return
import numpy
import matplotlib
import matplotlib.pyplot as plt
import scipy.special
import sklearn
import sys
import sklearn.datasets
import pylab
from Generative_Models import MultivariateGaussianClassifier

def mcol(v):
    return v.reshape((v.size,1))

def mrow(v):
    return v.reshape((1,v.size))

def logpdf_GAU_ND(x,mu,C):
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]* numpy.log(numpy.pi*2) + 0.5*numpy.linalg.slogdet(P)[1] - 0.5 * (numpy.dot(P, (x-mu)) *(x-mu)).sum(0)

def ML_GAU(D):
    mu = mcol(D.mean(1))
    C = numpy.dot(D-mu, (D-mu).T)/float(D.shape[1])
    return mu, C

def compute_llrs(DTR, DTE, LTR, LTE):
    h = {}

    for lab in [0,1]:
    
      mu, C = ML_GAU(DTR[:, LTR==lab]) 
      h[lab] = (mu, C)
    
    llrs = numpy.zeros((2,DTE.shape[1])) 

    for lab in [0,1]:
        mu, C = h[lab]
        llrs[lab, :] = numpy.exp(logpdf_GAU_ND(DTE,mu, C).ravel())
        
    llrs = numpy.log(llrs[1]/llrs[0])
    return llrs

def compute_min_DCF(llrs, Labels, pi, cfn, cfp):
    triplete = numpy.array([pi,cfn,cfp]) #pi, Cfn, Cfp
    thresholds = numpy.array(llrs)
    thresholds.sort()
    thresholds = numpy.concatenate([ numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf]) ])
    
    FPR = numpy.zeros(thresholds.size)
    FNR = numpy.zeros(thresholds.size)
    DCF_norm = numpy.zeros(thresholds.size)
    Bdummy1=triplete[0] * triplete[1] 
    Bdummy2=(1-triplete[0]) *triplete[2] 
    B_dummy = min(Bdummy1, Bdummy2)
    
    for idx, t in enumerate(thresholds):
        Pred = numpy.int32(llrs > t)
        Conf = numpy.zeros((2,2))
        for i in range(2):
            for j in range(2):
                Conf[i,j]= ((Pred==i) * (Labels == j)).sum()
        
        FPR[idx] = Conf[1,0] / (Conf[1,0]+ Conf[0,0])
        FNR[idx] = Conf[0,1] / (Conf[0,1]+ Conf[1,1])
        DCF_norm[idx] =  (Bdummy1*FNR[idx] + Bdummy2*FPR[idx]) / B_dummy
        
    DCF_min =  min(DCF_norm)
    return DCF_min

def compute_act_DCF(llrs, Labels, pi, cfn, cfp):
    triplete = numpy.array([pi,cfn,cfp]) #pi, Cfn, Cfp
    
    thread = (triplete[0]*triplete[1]) / ( ( 1- triplete[0])*triplete[2] ) 
    thread = -numpy.log(thread)
    
    LPred = llrs>thread
    
    Conf = numpy.zeros((2,2))
    for i in range(2):
        for j in range(2):
            Conf[i,j] = ((LPred==i) * (Labels == j)).sum()
            
    FPR = Conf[1,0] / (Conf[1,0]+ Conf[0,0]) 
    FNR = Conf[0,1] / (Conf[0,1]+ Conf[1,1]) 
    
    Bdummy1=triplete[0] * triplete[1] 
    Bdummy2=(1-triplete[0]) *triplete[2] 
    DCF =  Bdummy1*FNR+ Bdummy2*FPR
    B_dummy = min(Bdummy1, Bdummy2)
    DCF_norm = DCF/B_dummy
    return DCF_norm

def bayes_error_plot(pArray,llrs,Labels, minCost=False):
    y=[]
    for p in pArray:
        pi = 1.0/(1.0+numpy.exp(-p))
        if minCost:
            y.append(compute_min_DCF(llrs, Labels, pi, 1, 1))
        else: 
            y.append(compute_act_DCF(llrs, Labels, pi, 1, 1))

    return numpy.array(y)


def BayesDecision(DTR, LTR, DTE, LTE):
    
    Pred = MultivariateGaussianClassifier.MultivariateGaussianClassifier(DTR, LTR, DTE, LTE)
    
    #consufion matrix
    print("Campioni totali: %d" % LTE.shape[0])
    
    #per ogni campione ho Labels[i] che mi dice la sua etichetta reale e Pred[i] che è l'etichetta che stimo io con la probabilita a posteriori
    Conf = numpy.zeros((2,2))
    for i in range(2):
        for j in range(2):
            Conf[i,j] = ((Pred==i) * (LTE == j)).sum()
    
    print(Conf)
    print("Di questi: %d sono True Negative" % Conf[0,0])
    print("%d sono False Negative" % Conf[0,1])
    print("%d sono False Positive" % Conf[1,0])
    print("%d sono True Positive" % Conf[1,1])
    
    #We now focus on making optimal decisions when priors and costs are not uniform. In particolar for a binary task the matrix C of costs is:
    #C = [0, Cfn]
    #    [Cfp, 0]
    #triplete = (0.5, 1, 1) = #pi, Cfn, Cfp
    triplete = numpy.array([0.5,1,1]) 
    llrs = compute_llrs(DTR, DTE, LTR, LTE)
    
    thread = (triplete[0]*triplete[1]) / ( ( 1- triplete[0])*triplete[2] ) 
    thread = -numpy.log(thread)
    
    LPred = llrs>thread
    
    Conf = numpy.zeros((2,2))
    for i in range(2):
        for j in range(2):
            Conf[i,j] = ((LPred==i) * (LTE == j)).sum()
    print(Conf)
    
    #Plot della curca ROC
    
    thresholds = numpy.array(llrs)
    thresholds.sort()
    thresholds = numpy.concatenate([ numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf]) ])
    FPR = numpy.zeros(thresholds.size)
    TPR = numpy.zeros(thresholds.size)
    
    for idx, t in enumerate(thresholds):
        Pred = numpy.int32(llrs > t)
        Conf = numpy.zeros((2,2))
        for i in range(2):
            for j in range(2):
                Conf[i,j]= ((Pred==i) * (LTE == j)).sum()
        
        TPR[idx] = Conf[1,1] / (Conf[1,1]+Conf[0,1])
        FPR[idx] = Conf[1,0] / (Conf[1,0]+ Conf[0,0])
    
    pylab.plot(FPR, TPR)
    pylab.show()
    
    # act_DCF
    print(compute_act_DCF(llrs, LTE, 0.5, 1, 1))
    
    #min_DCF
    print( compute_min_DCF(llrs, LTE, 0.5, 1, 1) )
    
    #bayes error plot
    p = numpy.linspace(-3,3,21)
    pylab.plot(p, bayes_error_plot(p, llrs, LTE,minCost=False), color='r')
    pylab.plot(p, bayes_error_plot(p, llrs, LTE,minCost=True), color='b')
    pylab.ylim([0, 1.1])
    pylab.xlim([-3, 3])
    pylab.show()