from configure import Configure as conf
from cost_funcs import cost, cost_to_f
from guess import rotationtransition, translationtransition, centre_of_rotation_new, uniformlysample_I, centre_of_rotation_old , centre_of_rotation_walk, getrotationcentre_points
from repr import Repr
from numpy import random
from z3verifier import z3_verifier 
from print import initialized, statistics, z3statistics, invariantfound, timestatistics, prettyprint_samplepoints
import copy
from dnfs_and_transitions import RTI_to_LII, list3D_to_listof2Darrays
from timeit import default_timer as timer
from math import inf
from copy import deepcopy
import numpy as np
import sys

def randomlysamplelistoflists(l, costlists):
    ind = random.choice( len(l))
    return (l[ind], costlists[ind])


def get_best_neighbor(I, repr, cost_prev, samplepoints, c, d, og_costlist, spin, k1):
    currcost = cost_prev 
    possible_invs = [I]
    possible_costlists = [og_costlist]
    for i in range(d):
        for j in range(c):
            oldpredicate = I[i][j][:-2]
            oldconst = I[i][j][:-1]
            rotneighbors = repr.get_coeffneighbors(I[i][j][:-2])
            translation_neighbors = []
            for k in [-1,1]:
                J = I[i][j].copy()
                J[-1] = J[-1] + k
                translation_neighbors.append(J)


            # for transneighbor in translation_neighbors:
            #     Inew = deepcopy(I)
            #     Inew[i][j] =  transneighbor
            #     (newcost, costlist, spinI) = cost( list3D_to_listof2Darrays(Inew), samplepoints)
            #     # print(Inew, newcost) #Debug
            #     if (newcost < currcost):
            #         currcost = newcost
            #         possible_invs = [Inew]
            #         possible_costlists = [costlist]
            #     elif (newcost == currcost):
            #         possible_invs.append(Inew)
            #         possible_costlists.append(costlist)
            #     else:
            #         continue

            for rotneighbor in rotneighbors:
                Inew = deepcopy(I)
                # centreofrotation = centre_of_rotation_new(I[i][j], rotneighbor, spin, k1) #Uses the math worked out before
                # Trying to work this out!!
                filteredpoints = getrotationcentre_points(samplepoints, og_costlist) 
                centreofrotation = centre_of_rotation_old( oldpredicate , filteredpoints , rotneighbor) #Uses random/ Gaussian Sampling
                # centreofrotation = centre_of_rotation_walk(I[i][j], filteredpoints) #Uses random walk on hyperplane.
                const = round(np.dot(np.array(rotneighbor), np.array(centreofrotation)), 1)                 
                Inew[i][j] = rotneighbor + [-1, const]
                # print(I, centreofrotation, Inew) # Gives possible rotation transitions!s
                (newcost, costlist, spinI) = cost( list3D_to_listof2Darrays(Inew), samplepoints)
                if (newcost < currcost):
                    currcost = newcost
                    possible_invs = [Inew]
                    possible_costlists = [costlist]
                elif (newcost == currcost):
                    possible_invs.append(Inew)
                    possible_costlists.append(costlist)
                else:
                    continue

    
    return (possible_invs, currcost, possible_costlists)                

def hill_climbing(repr: Repr):
    tmax = repr.get_tmax()
    samplepoints = (repr.get_plus0(), repr.get_minus0(), repr.get_ICE0())
    initialized()
    z3_callcount = 0

    I = uniformlysample_I( repr.get_coeffvertices(), repr.get_k1(), repr.get_c(), repr.get_d(), repr.get_n())
    #Deterministic X_0:
    # I = [[ [-1,1, -1, 0], [0,-1, -1, 0] ]]

    
    LII = list3D_to_listof2Darrays(I)
    (costI, costlist, spinI) = cost(LII, samplepoints)  #spin = |-| - |+|
    prettyprint_samplepoints(samplepoints, "Selected-Points", "\t")
    print("\n")
    statistics(0, 1, I, -1, costI, 0, 0, costlist, -1 ) 

    while (1):
        for t in range(1,tmax + 1):
            if (costI == 0): #Put this in the start, because if by some magic we guess the first invariant in the first go, we dont want to change
                break  
            (besttransitions, costI, costlists) = get_best_neighbor(I, repr, costI, samplepoints, repr.get_c(), repr.get_d(), costlist, spinI, repr.get_k1())
            
            if (besttransitions == [I]):
                print("LOCAL MAXIMA wrt our transition guesses!!")

            (I, costlist) = randomlysamplelistoflists(besttransitions, costlists)
            
            # print(besttransitions, I, costI, costlist)
            statistics(t, 1, I, -1, costI, 0, 0, costlist, -1  )
        
        (z3_correct, cex) = z3_verifier(repr.get_P_z3expr(), repr.get_B_z3expr(), repr.get_T_z3expr(), repr.get_Q_z3expr(), list3D_to_listof2Darrays(I) )
        z3_callcount = z3_callcount + 1
        z3statistics(z3_correct, samplepoints, cex, z3_callcount, (t == tmax))
        if (z3_correct):
            break        
        samplepoints = (samplepoints[0] + cex[0] , samplepoints[1] + cex[1], samplepoints[2] + cex[2])
        (costI, costlist, spinI) = cost(list3D_to_listof2Darrays(I), samplepoints)  #spin = |-| - |+|
        
        statistics(0, 1, I, -1, costI, 1, 0, costlist, -1  )
    return            