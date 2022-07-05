
""" Imports.
"""
import input
from configure import Configure as conf
from cost_funcs import Cost
from dnfs_and_transitions import DNF_to_z3expr, DNF_to_z3expr_p
from guess import uniformlysampleLII, Guess, GuessStrategy, deg
from repr import Repr
import repr
from z3 import *
import numpy as np
from math import floor
from z3verifier import z3_verifier , DNF_to_z3expr

parameters = conf()

""" Main function. """

def metropolisHastings (repr: Repr):
    t = 0
    tmax = repr.get_tmax()
    samplepoints = (repr.get_plus0(), repr.get_minus0(), repr.get_ICE0())
    (I, deglist, cost, mincost) = uniformlysampleLII( repr.get_Dp(), repr.get_c(), repr.get_d(), repr.get_n(), samplepoints )
    while (1):
        while(t <= tmax):
            (I_new, deglist_new, cost_new, mincost_new) = randomwalktransition(I, deg, repr.get_Dp(), samplepoints)          
            a = min((1 - p) * ( deg(deglist) / deg(deglist_new) ) * (cost_new / cost) , 1)
            if (np.random.rand() <= a)          
                I = I_new
                cost = cost_new
                mincost = mincost_new
                deglist = deglist_new             
            else:
                continue
            if (mincost == 0):
                break  
        (correct, cex) = z3_verifier(repr.get_P_z3expr(), repr.get_B_z3expr(), repr.get_T_z3expr(), repr.get_Q_z3expr(), DNF_to_z3expr(I, primed = 0) )           
        if (correct):
            break
        samplepoints[0] = samplepoints[0] + cex[0]
        samplepoints[1] = samplepoints[1] + cex[1]
        samplepoints[2] = samplepoints[2] + cex[2]
    
    return (I, cost, t)




