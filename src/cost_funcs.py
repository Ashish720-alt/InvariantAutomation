
""" Cost Functions.
"""

import numpy as np
from dnfs_and_transitions import dnfnegation
from configure import Configure as conf
from copy import deepcopy
from math import inf, sqrt
from scipy.optimize import minimize, LinearConstraint

# This is a good normalization function for our context as it gives higher weight to smaller positive values (which are frequent datapoints), and not 
# that high weight to larger positive values (which are less frequent datapoints)
def normalizationfn(x, K):
    return K*(x/(1.0 + x))

def negationLIpredicate(p):
    return (dnfnegation( [np.array( [p], ndmin = 2 )] ))[0][0]

def LIPptdistance(p, pt):
    magnitude = sqrt(sum(i*i for i in p[:-2]))
    return max( (sum(p[:-2]* pt) - p[-1])/(magnitude)  , 0.0)

def LIccptdistance(cc, pt):
    rv = 0.0
    for p in cc:
        rv = rv + LIPptdistance(p, pt)
    return rv

# Assumes cc has only <= operator.
# Note that this ONLY gives the correct answer upto usually 12 decimal places, as approximation methods are used to solve ILP problems.
# Way too slow -- not an option, about 1/2 second per iteration for MCMC for c = 2,d=1 and 2-3 seconds for GD.
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
        rv = min(rv, LIccptdistance(cc, pt)) #Average of individual distances Variant
        # rv = min(rv, LIccptdistance_ILP(cc, pt)) #ILP Variant
    return rv

def costplus(I, pluspoints):
    spin = 0
    rv = 0.0
    rvlist = [] 
    for plus in pluspoints:
        cost = LIDNFptdistance(I, plus)
        if (cost > 0):
            spin = spin - 1
        rv = rv + cost
        rvlist.append(cost)
    return (rv, rvlist, spin)

def costminus(I, minuspoints):
    spin = 0
    negI = dnfnegation(I)
    rv = 0.0
    rvlist = [] 
    for minus in minuspoints:
        cost = LIDNFptdistance(negI, minus)
        if (cost > 0):
            spin = spin + 1
        rv = rv + cost
        rvlist.append(cost)
    return (rv, rvlist, spin)    

def costICE(I, ICEpoints):
    spin = 0
    negI = dnfnegation(I) 
    rv = 0.0
    rvlist = [] 
    for ICE in ICEpoints:
        cost1 = LIDNFptdistance(negI, ICE[0])
        cost2 = LIDNFptdistance(I, ICE[1])
        cost = min(cost1, cost2)
        if (cost > 0):
            if (cost == cost1):
                spin = spin + 1
            else:
                spin = spin - 1
        rv = rv + cost
        rvlist.append(cost)
    return (rv, rvlist, spin)

def U(r):
    return 1.0

def cost_to_f(totalcost):
    exp = normalizationfn(totalcost, conf.beta0)
    den = U(totalcost)
    return conf.alpha * (conf.gamma **(-exp) / den)


def cost(I, tupleofpoints):
    (cost_plus, l1, spinplus) = costplus(I, tupleofpoints[0])
    (cost_minus, l2, spinminus) = costminus(I, tupleofpoints[1])
    (cost_ICE, l3, spinICE) = costICE(I, tupleofpoints[2])
    return (cost_plus + cost_minus + cost_ICE , l1 + l2 + l3 , spinplus + spinminus + spinICE) # Debugging





