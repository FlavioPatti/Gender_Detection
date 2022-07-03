import numpy
import matplotlib.pyplot as plt
import scipy.optimize

def mcol(v):
    return v.reshape((v.size,1))

def mrow(v):
    return v.reshape((1,v.size))

def train_SVM_linear(DTR, LTR, DTE, LTE, C, K):
    DTREXT = numpy.vstack([DTR, K* numpy.ones((1,DTR.shape[1] ))])
    
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    
    H = numpy.dot(DTREXT.T, DTREXT)
    H = mcol(Z)* mrow(Z) * H
    
    def JDual(alpha): #dual SVM
        Ha = numpy.dot(H,mcol(alpha))
        aHa = numpy.dot(mrow(alpha), Ha)
        a1 = alpha.sum()
        return -0.5* aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size) 
                                        #derivata rispetto a alpha(opzionale perchè poi la funzione lo calcola automaticamente)
    def LDual(alpha): 
        loss, grad = JDual(alpha)
        return -loss, -grad 
    #il dual problem prevede di calcolare la soluzione massimizzando rispetto alpha la loss
    #utilizzando fmin_l_bfgs_b vogliamo minimizzare la nostra funzione obiettivo che otteniamo semplicemente passando alla funzione -loss
    
    def JPrimal(w): #primal SVM
        S = numpy.dot(mrow(w), DTREXT) #mrow(w) = w*T
        loss = numpy.maximum(numpy.zeros(S.shape), 1-Z*S).sum()
        return 0.5 *numpy.linalg.norm(w)**2 + C * loss
    
    
# LDual: la funzione che vogliamo minimizzare.
# x0: il valore iniziale per l'algoritmo. [0 ,.., 0]
#restituisce una tupla con tre valori alphaStar, _x, _y
# alphaStar è la posizione stimata del minimo
 
    alphaStar, _x, _y, = scipy.optimize.fmin_l_bfgs_b( #calcolo la soluzione dalla dual perchè è differenziabile mentre la primal no
        LDual, numpy.zeros(DTR.shape[1]), bounds = [(0,C)] * DTR.shape[1], factr = 0.0, maxiter = 100000, maxfun = 100000
    )
    
    #una volta che ho calcolato alphaStar, primal e dual problem sono legate da w = Sommatoria: alphaStar*z*x
    wStar = numpy.dot(DTREXT, mcol(alphaStar)*mcol(Z))
    
    DTEEXT = numpy.vstack([DTE, K* numpy.ones((1,DTE.shape[1] ))])    
     
    S1 = numpy.dot(wStar.T, DTEEXT);
        
    LP = S1>0
    LP = numpy.hstack(mrow(LP))
        
    accuracy = 0
    for i in range(LTE.shape[0]):
        if(LP[i]==LTE[i]):
            accuracy+=1
                
    accuracy = accuracy / DTE.shape[1]
    errore = 1 - accuracy
    
    print("Errore: ", 100*errore)
    
    return S1

    