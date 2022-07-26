from configure import Configure as conf
from cost_funcs import f
from guess import randomlysamplelistoflists, allowedrotations, uniformlysampleRTI, translationneighbors, translationdegree, rotationdegree, rotationtransition, translationtransition, ischange, get_index, isrotationchange
from repr import Repr
from numpy import random
from z3verifier import z3_verifier 
from print import initialized, statistics, z3statistics, invariantfound, timestatistics
import copy
from dnfs_and_transitions import RTI_to_LII
from timeit import default_timer as timer
from math import inf
from copy import deepcopy

def get_best_neighbor(I, repr, cost, samplepoints, beta, c, d):
    currcost = cost 
    possible_invs = [deepcopy(I)]
    index = (0,0,0,0)
    for i in range(d):
        for j in range(c):
            oldtranslationpred = I[i][j][1]
            oldrotationpred = I[i][j][0]
            rotationneighbors = allowedrotations(repr.get_coeffneighbors(oldrotationpred), oldtranslationpred, repr.get_k1() )
            translation_neighbors = translationneighbors(oldtranslationpred, oldrotationpred, repr.get_k1()) 
            
            for transneighbor in translation_neighbors:
                Inew = deepcopy(I)
                Inew[i][j][1] =  transneighbor
                (_, newcost) = f(RTI_to_LII(Inew), samplepoints, beta)
                # print(newcost) #
                if (newcost < currcost):
                    currcost = newcost
                    possible_invs = [Inew]
                elif (newcost == currcost):
                    possible_invs.append(Inew)
                else:
                    continue

            for rotneighbor in rotationneighbors:
                Inew = deepcopy(I)
                Inew[i][j][0] =  rotneighbor
                (_, newcost) = f(RTI_to_LII(Inew), samplepoints, beta)
                # print(newcost) #
                if (newcost < currcost):
                    currcost = newcost
                    possible_invs = [Inew]
                elif (newcost == currcost):
                    possible_invs.append(Inew)
                else:
                    continue

    
    return (possible_invs, currcost)                

def hill_climbing(repr: Repr):
    tmax = repr.get_tmax()
    samplepoints = (repr.get_plus0(), repr.get_minus0(), repr.get_ICE0())
    initialized()
    z3_callcount = 0
    beta = conf.beta0/(repr.get_c() * ( len(samplepoints[0]) + len(samplepoints[1]) + len(samplepoints[2])) * repr.get_theta0() )  
    I = uniformlysampleRTI( repr.get_coeffvertices(), repr.get_k1(), repr.get_c(), repr.get_d(), repr.get_n())
    LII = RTI_to_LII(I)
    (_, costI) = f(LII, samplepoints, beta)    
    while (1):
        for t in range(1,tmax + 1):
            if (costI == 0): #Put this in the start, because if by some magic we guess the first invariant in the first go, we dont want to change
                break  
            (besttransitions, costI) = get_best_neighbor(I, repr, costI, samplepoints, beta, repr.get_c(), repr.get_d())
            #print(besttransitions)
            I = randomlysamplelistoflists(besttransitions)
            statistics(t, 1, I, "-", costI, 1, 0 )
        
        (z3_correct, cex) = z3_verifier(repr.get_P_z3expr(), repr.get_B_z3expr(), repr.get_T_z3expr(), repr.get_Q_z3expr(), RTI_to_LII(I) )
        z3statistics(z3_correct, samplepoints, cex, z3_callcount, (t == tmax))
        if (z3_correct):
            break        
        samplepoints = (samplepoints[0] + cex[0] , samplepoints[1] + cex[1], samplepoints[2] + cex[2])
        (fI, costI) = f( RTI_to_LII(I), samplepoints, beta ) #samplepoints has changed, so cost and f changes
        beta = conf.beta0/(repr.get_c() * ( len(samplepoints[0]) + len(samplepoints[1]) + len(samplepoints[2])) * repr.get_theta0() )
        statistics(t, 1, I, "-", costI, 1, 0 )
    return            