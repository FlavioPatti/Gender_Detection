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


#la versione di Naive Bayes è semplicemente un classificatore gaussiano in cui le matrici di covarianza C per ogni classe sono diagonali.
#possiamo adattare il codice MVG semplicemente azzerando gli elementi fuori diagonale della soluzione MVG ML.
#Questo può essere fatto, ad esempio, moltiplicando element wise la soluzione MVG ML con la matrice di identità
def ML_GAU(D):
    mu = vcol(D.mean(1))
    C = numpy.dot(D-mu, (D-mu).T)/float(D.shape[1]) #4x4
    I = numpy.eye(D.shape[0])
    C_naive_bayes = C * I #element wise
    return mu, C_naive_bayes

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


def NaiveBayesClassifier(DTrain, LTrain, DTest, LTest): 

    h = {}

    for lab in [0,1]:
        mu, C = ML_GAU(DTrain[:, LTrain==lab]) 
        h[lab] = (mu, C)
    
    SJoint = numpy.zeros((2,DTest.shape[1])) 
    logSJoint = numpy.zeros((2, DTest.shape[1]))
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
    print("Accuracy for the Naive Bayes Classifier: ")
    print("%.2f" % (100*accuracy) )

    errore = 1 - accuracy
    print("Error for the Naive Bayes Classifier: ")
    print("%.2f\n" % (100* errore) )
    
    #for minDCF
    llrs = compute_llrs(DTrain, DTest, LTrain, LTest)
    return llrs
 
 