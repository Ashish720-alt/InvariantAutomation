
""" Cost Functions.
"""

import numpy as np
from dnfs_and_transitions import dnfnegation
from configure import Configure as conf
from copy import deepcopy
from math import inf, sqrt , e
from scipy.optimize import minimize, LinearConstraint

# This is a good normalization function for our context as it gives higher weight to smaller positive values (which are frequent datapoints), and not 
# that high weight to larger positive values (which are less frequent datapoints)
def normalizationfn(x, K):
    return K*(x/(1.0 + x))

def sigmoidfn(x, K):
    return 2*K*(1/(1 + np.exp(-x)) - 0.5)

def negationLIpredicate(p):
    return (dnfnegation( [np.array( [p], ndmin = 2 )] ))[0][0]

def distanceNormalizer(d):
    HP_K = 50.0
    HP_m = 1.0
    if (d <= HP_K):
        return (1.0*d)/HP_m
    else:
        return HP_K / (HP_m * ( 1 + e**(-(d - HP_K))) )
    


def LIPptdistance(p, pt):
    magnitude = sqrt(sum(i*i for i in p[:-2]))
    # return max( (sum(p[:-2]* pt) - p[-1])/(magnitude)  , 0.0) #CHANGE HERE
    return max( (sum(p[:-2]* pt) - p[-1])/(magnitude)  , 0.0) #Normalized distance


# def LIPptdistance(p, pt): #Uncomment 1 of the two
#     magnitude = sqrt(sum(i*i for i in p[:-2]))
#     return  (sum(p[:-2]* pt) - p[-1])/(magnitude) 


def LIccptdistance(cc, pt):
    rv = 0.0
    for p in cc:
        rv = rv + LIPptdistance(p, pt)
    return rv

# Assumes cc has only <= operator.
# Note that this ONLY gives the correct answer upto usually 12 decimal places, as approximation methods are used to solve ILP problems.
# Way too slow -- not an option, about 1/2 second per iteration for MCMC for c = 2,d=1 and 2-3 seconds for GD.
# def LIccptdistance_ILP(cc, pt):
#         n = len(cc[0]) - 2
#         A = np.concatenate(
#             [cc[:, :n], cc[:, n+1:]], axis=1)
#         return float(minimize(  
#             lambda x, pt: np.linalg.norm(x - pt),
#             x0 = np.zeros(n),
#             args=(pt,),
#             constraints=[LinearConstraint(A[:, :-1], lb = -np.inf, ub = A[:, -1])],
#         ).fun)
        


def LIDNFptdistance(dnf, pt):
    rv = inf
    for cc in dnf:
        rv =  distanceNormalizer(min(rv, LIccptdistance(cc, pt))) #Sum of individual distances Variant
        # rv = min(rv, LIccptdistance_ILP(cc, pt)) #ILP Variant
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


# def cost_to_f(totalcost, temp):
#     # exp = sigmoidfn(totalcost, conf.beta0) #Sigmoid variant
#     exp = normalizationfn(totalcost, conf.beta0) #Linear normalization
#     return conf.alpha * (conf.gamma **(-exp) / temp)


def cost(I, tupleofpoints):
    (cost_plus, l1) = costplus(I, tupleofpoints[0])
    (cost_minus, l2) = costminus(I, tupleofpoints[1])
    (cost_ICE, l3) = costICE(I, tupleofpoints[2])
    return (cost_plus/ len(l1) + cost_minus/ len(l2) + cost_ICE/len(l3) , l1 + l2 + l3 ) #Average dataset distance
    # return (cost_plus + cost_minus + cost_ICE , l1 + l2 + l3 )      #CHANGE HERE






