
""" Imports.
"""
from configure import Configure as conf
from cost_funcs import cost
from guess import uniformlysampleLII, randomwalktransition, deg
from repr import Repr
from numpy import random
from z3verifier import z3_verifier 
from print import initialized, statistics, z3statistics


""" Main function. """

def metropolisHastings (repr: Repr):
    tmax = repr.get_tmax()
    samplepoints = (repr.get_plus0(), repr.get_minus0(), repr.get_ICE0())
    initialized()
    (I, deglistI, costI, mincostI, mincosttupleI) = uniformlysampleLII( repr.get_Dp(), repr.get_c(), repr.get_d(), repr.get_n(), samplepoints )
    while (1):
        t = 0
        while(t <= tmax):
            (I_new, deglist_new, cost_new, mincost_new, mincosttuple_new) = randomwalktransition(I, deglistI, repr.get_Dp(), samplepoints, mincosttupleI)          
            a = min( ((deg(deglistI) * cost_new) / deg(deglist_new)) / costI , 1) #Make sure we don't underapproximate to 0
            if (random.rand() <= (1 - conf.p) *a):          
                (I, deglistI, costI, mincostI, mincosttupleI) = (I_new, deglist_new, cost_new, mincost_new, mincosttuple_new)       
            else:
                continue
            statistics(t, I, costI, mincostI)
            if (mincostI == 0):
                break  
        (correct, cex) = z3_verifier(repr.get_P_z3expr(), repr.get_B_z3expr(), repr.get_T_z3expr(), repr.get_Q_z3expr(), I )           
        if (correct):
            break
        z3statistics(correct, samplepoints, cex)
        samplepoints = (samplepoints[0] + cex[0] , samplepoints[1] + cex[1], samplepoints[2] + cex[2])
        (costI, mincostI, mincosttupleI) = cost(I, samplepoints)
    return (I, cost, t)


# Testing
from dnfs_and_transitions import dnfTrue
import numpy as np
from repr import genLItransitionrel

P = [np.array([[1, 0, 0]])]
B = [np.array([[1, -1, 5]])]
Q = [np.array([[1, 0, 6]])]
T = genLItransitionrel(B, ( [np.array([[1, 1], [0, 1]])] , dnfTrue(1) ) ) 
A = Repr(P, B, T, Q)
metropolisHastings(A)

