
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
    rv = 0.0
    rvlist = [] # Debugging
    for plus in pluspoints:
        rv = rv + LIDNFptdistance(I, plus)/ len(pluspoints)
        rvlist.append(LIDNFptdistance(I, plus)/ len(pluspoints))
    return (rv, rvlist)

def costminus(I, minuspoints):
    negI = dnfnegation(I)
    rv = 0.0
    rvlist = [] # Debugging
    for minus in minuspoints:
        rv = rv + LIDNFptdistance(negI, minus)/ len(minuspoints)
        rvlist.append(LIDNFptdistance(negI, minus)/ len(minuspoints))
    return (rv, rvlist)    

def costICE(I, ICEpoints):
    negI = dnfnegation(I) 
    rv = 0.0
    rvlist = [] # Debugging
    for ICE in ICEpoints:
        rv = rv + min( LIDNFptdistance(negI, ICE[0]), LIDNFptdistance(I, ICE[1]))/ len(ICEpoints)
        rvlist.append(min( LIDNFptdistance(negI, ICE[0]), LIDNFptdistance(I, ICE[1]))/ len(ICEpoints))
    return (rv, rvlist)

def U(r):
    return 1.0

def cost_to_f(totalcost, beta  ):
    return conf.alpha * (conf.gamma**(-beta * totalcost) / U(totalcost))


def cost(I, tupleofpoints, beta  ):
    (cost_plus, l1) = costplus(I, tupleofpoints[0])
    (cost_minus, l2) = costminus(I, tupleofpoints[1])
    (cost_ICE, l3) = costICE(I, tupleofpoints[2])
    return (cost_plus + cost_minus + cost_ICE , l1 + l2 + l3 ) # Debugging





