
""" Imports.
"""
from configure import Configure as conf
from cost_funcs import cost
from guess import uniformlysampleLII, randomwalktransition, deg
from repr import Repr
from numpy import random
from z3verifier import z3_verifier 
from print import initialized, statistics, z3statistics, invariantfound


""" Main function. """

def metropolisHastings (repr: Repr):
    tmax = repr.get_tmax()
    samplepoints = (repr.get_plus0(), repr.get_minus0(), repr.get_ICE0())
    initialized()
    (I, deglistI, costI, mincostI, mincosttupleI) = uniformlysampleLII( repr.get_Dp(), repr.get_c(), repr.get_d(), repr.get_n(), samplepoints )
    z3_callcount = 0
    while (1):
        for t in range(tmax):
            (I_new, deglist_new, cost_new, mincost_new, mincosttuple_new) = randomwalktransition(I, deglistI, repr.get_Dp(), samplepoints, mincosttupleI)          
            a = min( ((deg(deglistI) * cost_new) / deg(deglist_new)) / costI , 1) #Make sure we don't underapproximate to 0
            if (random.rand() <= (1 - conf.p) *a):          
                (I, deglistI, costI, mincostI, mincosttupleI) = (I_new, deglist_new, cost_new, mincost_new, mincosttuple_new)       
            else:
                continue
            statistics(t, I, costI, mincostI)
            if (mincostI == 0):
                break  
        (z3_correct, cex) = z3_verifier(repr.get_P_z3expr(), repr.get_B_z3expr(), repr.get_T_z3expr(), repr.get_Q_z3expr(), I )           
        z3_callcount = z3_callcount + 1
        z3statistics(z3_correct, samplepoints, cex, z3_callcount)
        if (z3_correct):
            break        
        samplepoints = (samplepoints[0] + cex[0] , samplepoints[1] + cex[1], samplepoints[2] + cex[2])
        (costI, mincostI, mincosttupleI) = cost(I, samplepoints)
    invariantfound(I)
    return (I, cost, t)




