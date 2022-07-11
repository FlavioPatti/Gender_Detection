import numpy 
import sklearn.datasets
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
    C = numpy.dot(D-mu, (D-mu).T)/float(D.shape[1]) #4x4
    I = numpy.eye(D.shape[0])
    C_naive_bayes = C * I 
    return mu, C_naive_bayes


def NaiveBayesClassifier(DTrain, LTrain, DTest): 
    """ Implementation of the  Naive Bayes Gaussian Classifier
        based on MVG version with log_densities,
        covariance matrixes are diagonal
        DTR and LTR are training data and labels
        DTE evaluation data 
        returns: the log-likelihood ratio
    """
    h = {}

    for lab in [0,1]:
        mu, C = ML_GAU(DTrain[:, LTrain==lab]) 
        h[lab] = (mu, C)
    
    llr = numpy.zeros((2, DTest.shape[1]))

    for lab in [0,1]:
        mu, C = h[lab]
     
        llr[lab, :] = logpdf_GAU_ND(DTest,mu, C).ravel()
    
    
    return llr[1]-llr[0]
 
 