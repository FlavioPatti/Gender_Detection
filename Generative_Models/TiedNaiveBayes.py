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

def TiedNaiveBayes(DTrain, LTrain, DTest, LTest):

    h = {} 

    #ct = sw = matrice di covarianza all'interno della classe
    Ct = compute_sw(DTrain, LTrain)
    I = numpy.eye(Ct.shape[0])
    Ct_naive_bayes = Ct * I 
    
    for lab in [0,1]:
    
        mu, C = ML_GAU(DTrain[:, LTrain==lab]) 
        h[lab] = (mu, Ct_naive_bayes)
    
    SJoint = numpy.zeros((3,DTest.shape[1])) 
    logSJoint = numpy.zeros((3, DTest.shape[1]))
    classPriors = [0.66/1.0, 0.33/1.0]

    for lab in [0,1]:
        mu, C = h[lab]
    
        SJoint[lab, :] = numpy.exp(logpdf_GAU_ND(DTest,mu, C).ravel()) * classPriors[lab] 
        logSJoint[lab, :] = logpdf_GAU_ND(DTest,mu, C).ravel() * classPriors[lab] 
    
    SMarginal = SJoint.sum(0) 
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    numpy.seterr(divide='ignore', invalid='ignore')
    Post1 = SJoint/vrow(SMarginal)  
    logPost = logSJoint - vrow(logSMarginal)
    Post2 = numpy.exp(logPost)

    numpy.seterr(divide='ignore', invalid='ignore')
    err = ((numpy.abs(Post2-Post1))/Post1)

    LPred1 = Post1.argmax(0) 
    Lpred2 = Post2.argmax(0)

    accuracy = (LTest==LPred1).sum()/DTest.shape[1]
    print("Accuracy for the Tied Naive Bayes Classifier: ")
    print("%.2f" % (100*accuracy) )

    errore = 1 - accuracy
    print("Error for the Tied Naive Bayes Classifier: ")
    print("%.2f\n" % (100* errore) )