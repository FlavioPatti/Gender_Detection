import numpy 
import matplotlib.pyplot as plt

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

"""USED ONLY IN THE TESTING PART! We decided, based on the quantity of data, to use a k-fold approach for training purposes"""
def split_db_2to1(D, L, param, seed=0):
    """ Split the dataset in two parts based on the param,
        first part will be used for model training, second part for testing
        D is the dataset, L the corresponding labels
        seed is set to 0 and it's used to randomize partitions
        returns:
        DTR_TRA = Dataset for training set
        LTR_TRA = Labels for training set
        DTR_TEST = Dataset for testing set
        LTR_TEST = Labels for testing set
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

