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



def compute_empirical_cov(D):
    mu = vcol(D.mean(1));
    DC = D-mu
    C = numpy.dot(DC,DC.T) / D.shape[1]
    return C 

def compute_sw(D,L): #within coviariance matrix
    SW = 0
    for i in [0,1]:
        SW+=  (L==i).sum() * compute_empirical_cov(D[:,L==i]) #calcolo la matrice di covarianza C come in PCA però per tutte le classi all'interno del dataset
        #(L==i).sum() = numero campioni per ogni classe = 50 
    return SW / D.shape[1]  


def TiedGaussianClassifier(DTrain,LTrain, DTest): 

    h = {}
    
    Ct = compute_sw(DTrain, LTrain)
    for lab in [0,1]:
    
        mu, C = ML_GAU(DTrain[:, LTrain==lab]) 
        h[lab] = (mu, Ct)

    llr = numpy.zeros((2, DTest.shape[1]))

    for lab in [0,1]:
        mu, C = h[lab]
     
        llr[lab, :] = logpdf_GAU_ND(DTest,mu, C).ravel()
    
    
    return llr[1]-llr[0]
 
    