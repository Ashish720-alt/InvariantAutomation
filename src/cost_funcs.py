
""" Cost Functions.
"""

import numpy as np
from dnfs_and_transitions import dnfnegation
from configure import Configure as conf
from copy import deepcopy
from math import inf, sqrt

def negationLIpredicate(p):
    return (dnfnegation( [np.array( [p], ndmin = 2 )] ))[0][0]

def LIPptdistance(p, pt):
    magnitude = sqrt(sum(i*i for i in p[:-2]))
    return max( (sum(p[:-2]* pt) - p[-1])/(magnitude*(conf.dspace_intmax - conf.dspace_intmin ))  , 0.0)

def LIccptdistance(cc, pt):
    rv = 0.0
    for p in cc:
        rv = rv + LIPptdistance(p, pt)
    return (rv/len(cc))

def LIDNFptdistance(dnf, pt):
    rv = inf
    for cc in dnf:
        rv = min(rv, LIccptdistance(cc, pt))
    return rv

def costplus(I, pluspoints):
    spin = 0
    rv = 0.0
    rvlist = [] # Debugging
    for plus in pluspoints:
        cost = LIDNFptdistance(I, plus)/ len(pluspoints)
        if (cost > 0):
            spin = spin - 1
        rv = rv + cost
        rvlist.append(cost)
    return (rv, rvlist, spin)

def costminus(I, minuspoints):
    spin = 0
    negI = dnfnegation(I)
    rv = 0.0
    rvlist = [] # Debugging
    for minus in minuspoints:
        cost = LIDNFptdistance(negI, minus)/ len(minuspoints)
        if (cost > 0):
            spin = spin + 1
        rv = rv + cost
        rvlist.append(cost)
    return (rv, rvlist, spin)    

def costICE(I, ICEpoints):
    spin = 0
    negI = dnfnegation(I) 
    rv = 0.0
    rvlist = [] # Debugging
    for ICE in ICEpoints:
        cost1 = LIDNFptdistance(negI, ICE[0])/ len(ICEpoints)
        cost2 = LIDNFptdistance(I, ICE[1])/ len(ICEpoints)
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

def cost_to_f(totalcost, beta  ):
    return conf.alpha * (conf.gamma**(-beta * totalcost) / U(totalcost))


def cost(I, tupleofpoints, beta  ):
    (cost_plus, l1, spinplus) = costplus(I, tupleofpoints[0])
    (cost_minus, l2, spinminus) = costminus(I, tupleofpoints[1])
    (cost_ICE, l3, spinICE) = costICE(I, tupleofpoints[2])
    return (cost_plus + cost_minus + cost_ICE , l1 + l2 + l3 , spinplus + spinminus + spinICE) # Debugging





