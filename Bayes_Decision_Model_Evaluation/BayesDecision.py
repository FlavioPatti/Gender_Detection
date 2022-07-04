import numpy

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


