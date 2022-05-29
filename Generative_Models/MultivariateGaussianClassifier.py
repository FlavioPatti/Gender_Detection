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

def MultivariateGaussianClassifier(DTrain, LTrain, DTest, LTest):
    
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
    SJoint = numpy.zeros((2,DTest.shape[1])) 
    logSJoint = numpy.zeros((2, DTest.shape[1]))
    #ipotizzo una probabilita a priori uniforme per le classi
    classPriors = [0.666/1.0, 0.333/1.0]

    for lab in [0,1]:
        mu, C = h[lab]
    
        SJoint[lab, :] = numpy.exp(logpdf_GAU_ND(DTest,mu, C).ravel()) * classPriors[lab] 
        logSJoint[lab, :] = logpdf_GAU_ND(DTest,mu, C).ravel() * classPriors[lab] 
    
    #2) calcolare le densita marginali: Σ(c)  fX,C (xt, c)
    #SMarginal: = somma di tutte le colonne di SJoint = somma su tutte le classi della probabilità congiunta (SJoint)
    SMarginal = SJoint.sum(0) 
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)
    
    #ignoro possibili warning legati a divisioni invalide
    numpy.seterr(divide='ignore', invalid='ignore')
    
    #3) calcolare la probabilita a posteriori: # P(C = c|X = xt) = fX,C (xt, c) / SOMMATORIA(c')  fX,C (xt, c')
    Post1 = SJoint/vrow(SMarginal) #matrice della probabilità a posteriori
    logPost = logSJoint - vrow(logSMarginal)
    Post2 = numpy.exp(logPost) 
    
    #le due matrici calcolate in decimale e in log sono pressochè uguali infatti differiscono per e^-.. dettato dal calcolo di floating point con precisione finita
    numpy.seterr(divide='ignore', invalid='ignore')
    err = ((numpy.abs(Post2-Post1))/Post1)
    

    #in ogni colonna di Post1 ho 2 probabilita a posteriori e assegno l'etichetta in base al valore maggiore
    #esempio: [1.00000000e+00 3.83816705e-24] => assegno la classe 0
    LPred1 = Post1.argmax(0) #argmax(0) = prende per ogni colonna l'indice di riga corrispondente all'argomento massimo
    Lpred2 = Post2.argmax(0)
    
    #calcolo accuratezza e errore per il classificatore

    accuracy = (LTest==LPred1).sum()/DTest.shape[1]
    print("Accuracy for the Multivariate Gaussian Classifier: ")
    print("%.2f" % (100*accuracy) )

    errore = 1 - accuracy
    print("Error for the Multivariate Gaussian Classifier: ")
    print("%.2f\n" % (100* errore) )
    
    return Lpred2
 