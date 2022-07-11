
""" Imports.
"""
from configure import Configure as conf
from cost_funcs import f
from guess import uniformlysampleLII, randomwalktransition, deg
from repr import Repr
from numpy import random
from z3verifier import z3_verifier 
from print import initialized, statistics, z3statistics, invariantfound
import copy

""" Main function. """

def metropolisHastings (repr: Repr):
    tmax = repr.get_tmax()
    samplepoints = (repr.get_plus0(), repr.get_minus0(), repr.get_ICE0())
    initialized()
    beta = conf.beta0/(repr.get_c() * ( len(samplepoints[0]) + len(samplepoints[1]) + len(samplepoints[2])) * repr.get_theta0() )
    (I, deglistI, fI, costI, costtupleI) = uniformlysampleLII( repr.get_Dp(), repr.get_c(), repr.get_d(), repr.get_n(), samplepoints, beta  )
    statistics(0, I, fI, costI, 0, 0 )
    z3_callcount = 0
    while (1):
        for t in range(1,tmax + 1):
            (I_new, deglist_new, f_new, cost_new, costtuple_new) = randomwalktransition(I, deglistI, repr.get_Dp(), samplepoints, costtupleI, beta )          
            descent = 1 if (cost_new > costI) else 0 
            a = min( ((deg(deglistI) * f_new) / deg(deglist_new)) / fI , 1) #Make sure we don't underapproximate to 0
            if (random.rand() <= (1 - conf.p) *a):          
                (I, deglistI, fI, costI, costtupleI) = (I_new, deglist_new, f_new, cost_new, costtuple_new)   
                statistics(t, I_new, f_new, cost_new, descent, 0 )    
            else:
                statistics(t, I_new, f_new, cost_new, descent, 1 )
                continue
            if (costI == 0):
                break  
        (z3_correct, cex) = z3_verifier(repr.get_P_z3expr(), repr.get_B_z3expr(), repr.get_T_z3expr(), repr.get_Q_z3expr(), I )           
        z3_callcount = z3_callcount + 1
        z3statistics(z3_correct, samplepoints, cex, z3_callcount, (t == tmax))
        if (z3_correct):
            break        
        samplepoints = (samplepoints[0] + cex[0] , samplepoints[1] + cex[1], samplepoints[2] + cex[2])
        (fI, costI, costtupleI) = f(I, samplepoints, beta )
        beta = conf.beta0/(repr.get_c() * ( len(samplepoints[0]) + len(samplepoints[1]) + len(samplepoints[2])) * repr.get_theta0() )
    invariantfound(I)
    return (I, f, z3_callcount)




