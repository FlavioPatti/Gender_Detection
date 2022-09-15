import numpy 
import utilities as ut
    
def logpdf_GAU_ND(x,mu,C):
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]* numpy.log(numpy.pi*2) + 0.5*numpy.linalg.slogdet(P)[1] - 0.5 * (numpy.dot(P, (x-mu)) *(x-mu)).sum(0)

def ML_GAU(D):
    mu = ut.vcol(D.mean(1))
    C = numpy.dot(D-mu, (D-mu).T)/float(D.shape[1])
    return mu, C


def compute_empirical_cov(D):
    mu = ut.vcol(D.mean(1))
    DC = D-mu
    C = numpy.dot(DC,DC.T) / D.shape[1]
    return C 

def compute_sw(D,L): 
    SW = 0
    for i in [0,1]:
        SW+=  (L==i).sum() * compute_empirical_cov(D[:,L==i])
    return SW / D.shape[1]  

def TiedNaiveBayes(DTrain, LTrain, DTest):
    """ Implementation of the Tied Naive Bayes Classifier
        based on MVG version with log_densities
        DTR and LTR are training data and labels
        DTE are evaluation data
        returns: the log-likelihood ratio
    """

    h = {} 

    Ct = compute_sw(DTrain, LTrain)
    I = numpy.eye(Ct.shape[0])
    Ct_naive_bayes = Ct * I 
    
    for lab in [0,1]:
        mu = ut.vcol(DTrain[:, LTrain==lab].mean(1)) 
        h[lab] = (mu, Ct_naive_bayes)
    
    llr = numpy.zeros((2, DTest.shape[1]))

    for lab in [0,1]:
        mu, C_naive_bayes = h[lab]
     
        llr[lab, :] = logpdf_GAU_ND(DTest,mu, C_naive_bayes).ravel()
    
    
    return llr[1]-llr[0]
 
 