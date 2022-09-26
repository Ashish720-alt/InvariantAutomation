
""" Imports.
"""
from configure import Configure as conf
from cost_funcs import cost, cost_to_f
from guess import uniformlysample_I, rotationdegree, rotationtransition, translationtransition, ischange, get_index, isrotationchange
from repr import Repr
from numpy import random
from z3verifier import z3_verifier 
from print import initialized, statistics, z3statistics, invariantfound, timestatistics, prettyprint_samplepoints
import copy
from dnfs_and_transitions import RTI_to_LII, list3D_to_listof2Darrays
from timeit import default_timer as timer
from math import isnan



""" Main function. """

def metropolisHastings (repr: Repr):
    mcmc_time = 0
    total_iterations = 0
    z3_time = 0
    initialize_time = 0
    z3_callcount = 0

    initialize_start = timer()
    tmax = repr.get_tmax()
    samplepoints = (repr.get_plus0(), repr.get_minus0(), repr.get_ICE0())
    initialized()
    
    I = uniformlysample_I( repr.get_coeffvertices(), repr.get_k1(), repr.get_c(), repr.get_d(), repr.get_n())
    

    # I = [[ [-1,1, -1, 850] ]] #Translation
    # I = [[ [1,-1, -1, 850] ]] #Rotation
    
    LII = list3D_to_listof2Darrays(I)
    (costI, costlist, spinI) = cost(LII, samplepoints)  #spin = |-| - |+|
    fI = cost_to_f(costI)
    prettyprint_samplepoints(samplepoints, "Selected-Points", "\t")
    print("\n")
    statistics(0, 1, I, fI, costI, 0, 0, costlist, -1 ) 

    initialize_end = timer()
    initialize_time = initialize_time + (initialize_end - initialize_start)


    while (1):
        mcmc_start = timer()
        for t in range(1,tmax + 1):
            if (costI == 0): #Put this in the start, because if by some magic we guess the first invariant in the first go, we dont want to change
                break  
            is_change = ischange() 
            if (is_change):
                index = get_index(repr.get_d(), repr.get_c())
                oldpredicate = I[index[0]][index[1]]
                is_rotationchange = isrotationchange()
                if (is_rotationchange ): 
                    rotneighbors = repr.get_coeffneighbors(oldpredicate[:-2])
                    deg = rotationdegree(rotneighbors)
                    newpred = rotationtransition(oldpredicate, rotneighbors, spinI, repr.get_k1()) #Change the code here!
                    degnew = rotationdegree(repr.get_coeffneighbors(newpred[:-2]))
                    I[index[0]][index[1]] = newpred
                else:
                    newpred = translationtransition(oldpredicate) 
                    I[index[0]][index[1]] = newpred
                    (deg, degnew) = (2,2)
                    
                (costInew, costlist, spinInew) = cost(list3D_to_listof2Darrays(I), samplepoints)
                fInew = cost_to_f(costInew)
                if (costInew <= costI):
                    a = 1.0
                else:
                    if (fI == 0 or isnan(fI)):
                        if (fInew == 0 or isnan(fInew)):
                            r = 10.0 #???
                            a = min( ( costI * deg)/ (r*costInew * degnew) , 1.0)                        
                        else:
                            a = 1.0
                    else:
                        a = min( ( fInew * deg)/ (fI * degnew) , 1.0) #Make sure we don't underapproximate to 0
                if (random.rand() <=  a): 
                    reject = 0
                    descent = 1 if (costInew > costI) else 0 
                    (fI, costI, spinI) = (fInew, costInew, spinInew)
                    statistics(t, 1, I, fInew, costInew, descent, reject, costlist,a )                   
                else:
                    reject = 1
                    descent = 0
                    statistics(t, 1, I, fInew, costInew, descent, reject, costlist, a )
                    I[index[0]][index[1]] = oldpredicate
            else:
                statistics(t, 0, I, fI, costI, 0, 0, costlist, -1 )
            
        mcmc_end = timer()
        mcmc_time = mcmc_time + (mcmc_end - mcmc_start)
        total_iterations = total_iterations + t
        
        z3_start = timer()
        (z3_correct, cex) = z3_verifier(repr.get_P_z3expr(), repr.get_B_z3expr(), repr.get_T_z3expr(), repr.get_Q_z3expr(), list3D_to_listof2Darrays(I) )           
        z3_end = timer()
        z3_time = z3_time + (z3_end - z3_start)

        initialize_start = timer()
        z3_callcount = z3_callcount + 1
        z3statistics(z3_correct, samplepoints, cex, z3_callcount, (t == tmax))
        if (z3_correct):
            break        
        samplepoints = (samplepoints[0] + cex[0] , samplepoints[1] + cex[1], samplepoints[2] + cex[2])
        (costI, costlist, spinI) = cost( list3D_to_listof2Darrays(I), samplepoints) #samplepoints has changed, so cost and f changes for same invariant
        fI = cost_to_f(costI)
        statistics(0, 1, I, fI, costI, 0, 0, [], -1 )
        initialize_end = timer()
        initialize_time = initialize_time + (initialize_end - initialize_start)

    invariantfound(I)
    timestatistics(mcmc_time, total_iterations, z3_time, initialize_time, z3_callcount )

    return (I, fI, z3_callcount)




