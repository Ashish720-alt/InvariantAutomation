
""" Imports.
"""
from configure import Configure as conf
from cost_funcs import f
from guess import uniformlysampleRTI, translationneighbors, translationdegree, rotationdegree, rotationtransition, translationtransition, ischange, get_index, isrotationchange
from repr import Repr
from numpy import random
from z3verifier import z3_verifier 
from print import initialized, statistics, z3statistics, invariantfound, timestatistics, prettyprint_samplepoints
import copy
from dnfs_and_transitions import RTI_to_LII
from timeit import default_timer as timer




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
    beta = conf.beta0/(repr.get_c() * ( len(samplepoints[0]) + len(samplepoints[1]) + len(samplepoints[2])) * repr.get_theta0() )  
    I = uniformlysampleRTI( repr.get_coeffvertices(), repr.get_k1(), repr.get_c(), repr.get_d(), repr.get_n())
    LII = RTI_to_LII(I)
    (fI, costI, costlist) = f(LII, samplepoints, beta) #Debugging
    
    prettyprint_samplepoints(samplepoints, "Selected-Points", "\t")
    statistics(0, 1, I, fI, costI, 0, 0, costlist ) 

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
                oldtranslationpred =  I[index[0]][index[1]][1]
                oldrotationpred = I[index[0]][index[1]][0]
                is_rotationchange = isrotationchange()
                if (is_rotationchange ):
                    rotneighbors = repr.get_coeffneighbors(oldrotationpred)
                    deg = rotationdegree(rotneighbors)
                    (newrotationpred, degnew) = rotationtransition(rotneighbors, oldtranslationpred, repr.get_k1())
                    if (degnew == 0):
                        newrotationpred = oldrotationpred
                        degnew = deg
                    I[index[0]][index[1]][0] = newrotationpred
                else:
                    deg = translationdegree(  oldtranslationpred , oldrotationpred , repr.get_k1())
                    (newtranspred, degnew) = translationtransition(oldtranslationpred, oldrotationpred, repr.get_k1())
                    I[index[0]][index[1]][1] = newtranspred
                (fInew, costInew, costlist) = f(RTI_to_LII(I), samplepoints, beta)
                a = min( fInew * deg/ fI * degnew , 1) #Make sure we don't underapproximate to 0
                if (random.rand() <=  a): 
                    reject = 0
                    descent = 1 if (costInew > costI) else 0 
                    (fI, costI) = (fInew, costInew)
                    statistics(t, 1, I, fInew, costInew, descent, reject, costlist )                   
                else:
                    reject = 1
                    descent = 0
                    statistics(t, 1, I, fInew, costInew, descent, reject, costlist )
                    if (is_rotationchange):
                        I[index[0]][index[1]][0] = oldrotationpred
                    else:
                        I[index[0]][index[1]][1] = oldtranspred
            else:
                statistics(t, 0, I, fI, costI, 0, 0, costlist )
            
        mcmc_end = timer()
        mcmc_time = mcmc_time + (mcmc_end - mcmc_start)
        total_iterations = total_iterations + t
        
        z3_start = timer()
        (z3_correct, cex) = z3_verifier(repr.get_P_z3expr(), repr.get_B_z3expr(), repr.get_T_z3expr(), repr.get_Q_z3expr(), RTI_to_LII(I) )           
        z3_end = timer()
        z3_time = z3_time + (z3_end - z3_start)

        initialize_start = timer()
        z3_callcount = z3_callcount + 1
        z3statistics(z3_correct, samplepoints, cex, z3_callcount, (t == tmax))
        if (z3_correct):
            break        
        samplepoints = (samplepoints[0] + cex[0] , samplepoints[1] + cex[1], samplepoints[2] + cex[2])
        (fI, costI) = f( RTI_to_LII(I), samplepoints, beta ) #samplepoints has changed, so cost and f changes for same invariant
        beta = conf.beta0/(repr.get_c() * ( len(samplepoints[0]) + len(samplepoints[1]) + len(samplepoints[2])) * repr.get_theta0() )
        statistics(0, 1, I, fI, costI, 0, 0 )
        initialize_end = timer()
        initialize_time = initialize_time + (initialize_end - initialize_start)

    invariantfound(I)
    timestatistics(mcmc_time, total_iterations, z3_time, initialize_time, z3_callcount )

    return (I, fI, z3_callcount)




