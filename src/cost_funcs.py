
""" Cost Functions. """

import numpy as np
from dnfs_and_transitions import dnfnegation
from configure import Configure as conf
from math import inf, sqrt , e


def negationLIpredicate(p):
    return (dnfnegation( [np.array( [p], ndmin = 2 )] ))[0][0]


def sigmoidfn(x):
    return 1/(1 + e**(-x)) 
        
def distanceNormalizer(d):
    if (d <= conf.costnormalizer_K):
        return d / conf.costnormalizer_m
    else:
        return (conf.costnormalizer_K / conf.costnormalizer_m) * sigmoidfn(d - conf.costnormalizer_K)
    
def distanceNormalizer2(x, K):
    return conf.costnormalizer_K *(x/(1.0 + x))


def LIPptdistance(p, pt):
    magnitude = sqrt(sum(i*i for i in p[:-2]))
    return max( (sum(p[:-2]* pt) - p[-1])/(magnitude)  , 0.0) 


def LIccptdistance(cc, pt):
    rv = 0.0
    for p in cc:
        rv = rv + LIPptdistance(p, pt)
    return rv

from scipy.optimize import minimize, LinearConstraint
''' (1) Assumes cc has only <= operator.
(2) Gives the correct answer upto usually 12 decimal places, as approximation methods are used to solve ILP problems.
(3) Way too slow -- not an option, about 1/2 second per iteration for MCMC for c = 2,d=1 and 2-3 seconds for GD. '''
def LIccptdistance_ILP(cc, pt):
        n = len(cc[0]) - 2
        A = np.concatenate(
            [cc[:, :n], cc[:, n+1:]], axis=1)
        return float(minimize(  
            lambda x, pt: np.linalg.norm(x - pt),
            x0 = np.zeros(n),
            args=(pt,),
            constraints=[LinearConstraint(A[:, :-1], lb = -np.inf, ub = A[:, -1])],
        ).fun)


def LIDNFptdistance(dnf, pt):
    rv = inf
    for cc in dnf:
        if (conf.COST_ILP == conf.ON):
            rv = min(rv, LIccptdistance_ILP(cc, pt)) 
        elif (conf.COST_DISTANCE_UNNORMALIZED == conf.ON):
            rv =  min(rv, LIccptdistance(cc, pt))
        elif (conf.COST_DISTANCE_NORMALIZED == conf.ON):
            rv =  distanceNormalizer(min(rv, LIccptdistance(cc, pt))) 
        else:
            raise Exception("conf.py Error: Exactly one of COST_ILP, COST_DISTANCE_UNNORMALIZED or COST_DISTANCE_NORMALIZED must be ON.")
    return rv

def costplus(I, pluspoints):
    rv = 0.0
    rvlist = [] 
    for plus in pluspoints:
        cost = LIDNFptdistance(I, plus)
        rv = rv + cost
        rvlist.append(cost)
    return (rv, rvlist)

def costminus(I, minuspoints):
    negI = dnfnegation(I)
    rv = 0.0
    rvlist = [] 
    for minus in minuspoints:
        cost = LIDNFptdistance(negI, minus)
        rv = rv + cost
        rvlist.append(cost)
    return (rv, rvlist)    

def costICE(I, ICEpoints):
    negI = dnfnegation(I) 
    rv = 0.0
    rvlist = [] 
    for ICE in ICEpoints:
        cost1 = LIDNFptdistance(negI, ICE[0])
        cost2 = LIDNFptdistance(I, ICE[1])
        cost = min(cost1, cost2)
        rv = rv + cost
        rvlist.append(cost)
    return (rv, rvlist)


def cost(I, tupleofpoints):
    def norm_avg(sum, N):
        if N == 0:
            return 0
        else:
            return sum / N
    
    (cost_plus, l1) = costplus(I, tupleofpoints[0])
    (cost_minus, l2) = costminus(I, tupleofpoints[1])
    (cost_ICE, l3) = costICE(I, tupleofpoints[2])
    if (conf.COST_DATASET_NORMALIZER == conf.ON):
        return (norm_avg(cost_plus, len(l1)) + norm_avg(cost_minus, len(l2)) + norm_avg(cost_ICE, len(l3)) , l1 + l2 + l3 ) 
    else:
        return (cost_plus + cost_minus + cost_ICE , l1 + l2 + l3 )  






