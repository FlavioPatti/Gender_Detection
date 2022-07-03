import numpy 
import scipy.special

def vcol(v):
    return v.reshape((v.size,1))

def vrow(v):
    return v.reshape((1,v.size))  

def logpdf_GAU_ND(x,mu,C):
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]* numpy.log(numpy.pi*2) + 0.5*numpy.linalg.slogdet(P)[1] - 0.5 * (numpy.dot(P, (x-mu)) *(x-mu)).sum(0)

def ML_GAU(D):
    mu = vcol(D.mean(1))
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

def MultivariateGaussianClassifier(DTrain, LTrain, DTest):
    
    #Multivariate Gaussian Classifier (MVG): il classificatore presuppone che i campioni di ciascuna classe c ∈ {0, 1} 
    #possano essere modellati come campioni di una distribuzione gaussiana multivariata con media e matrice di covarianza dipendenti dalla classe(indipendenti tra loro)
    
    #Classificazione in base alla più alta probabilità a posteriori assegnata a ciascuna classe P(c|x) calcolata in 3 step:

    #1) calcolo della likelihood condizionata ovvero la likelihood fX|C (xt|c) = N (xt|µ)t|µ*,Σ*) * la probabilita a priori Pc(c) = 1/2 per le due classi


    h = {} #hash table in cui la chiave è la classe(c0,c1)
           #h[0] = (mu0, c0) = µ1,Σ1
           #h[1] = (mu1, c1) = µ2,Σ2

    for lab in [0,1]:
    #calcolo per ogni classe la media e la matrice di covarianza
        mu, C = ML_GAU(DTrain[:, LTrain==lab]) #unione della def compute_empirical_mean() e compute_empirical_cov() che ritorna entrambi i valori
        h[lab] = (mu, C)
    
    #matrice delle joint density[2,50]
    #ogni riga della matrice corrisponde a una classe, e contiene le log-likelihood condizionali per tutti i campioni per quella classe
    llr = numpy.zeros((2, DTest.shape[1]))

    for lab in [0,1]:
        mu, C = h[lab]
    
        llr[lab, :] = logpdf_GAU_ND(DTest,mu, C).ravel() 
    

    return llr[1]-llr[0]
 