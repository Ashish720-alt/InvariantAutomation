""" Imports.
"""
from configure import Configure as conf
from cost_funcs import cost, cost_to_f
from guess import uniformlysample_I, rotationdegree, rotationtransition, translationtransition, get_index, isrotationchange, getrotationcentre_points
from repr import Repr
from numpy import random
from z3verifier import z3_verifier 
from print import initialized, statistics, z3statistics, invariantfound, timestatistics, prettyprint_samplepoints
import copy
from dnfs_and_transitions import RTI_to_LII, list3D_to_listof2Darrays, dnfconjunction
from timeit import default_timer as timer
from math import isnan, log



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
    initialized( repr.get_affineSubspace() , repr.get_Var())
    


    I_guess = uniformlysample_I( repr.get_coeffvertices(), repr.get_k1(), repr.get_c(), repr.get_d(), repr.get_n())

    # I_guess = [ [ [-1, 0, -1, -50], [1, -1, -1, 0], [-1, 1, -1, 0], [1, 0, -1, 100]  ] , [ [1, 0, -1, 50], [1, 0, -1, 100], [0, 1, -1, 50], [0, -1, -1, -50] ] ]

    # I_guess = [[ [-1, 0, 0, -2, 0] ] , [ [0, -1, 0, -2, 0] ] , [ [0, 0, -1, -2, 0] ]] #Deterministic start


    LII = dnfconjunction( list3D_to_listof2Darrays(I_guess), repr.get_affineSubspace() , 0)
    (costI, costlist, spinI) = cost(LII, samplepoints)  #spin = |-| - |+|
    temp = conf.temp_C/log(2)
    fI = cost_to_f(costI, temp)
    prettyprint_samplepoints(samplepoints, "Selected-Points", "\t")
    print("\n")
    statistics(0, I_guess, fI, costI, 0, 0, costlist, -1, repr.get_Var() ) 

    initialize_end = timer()
    initialize_time = initialize_time + (initialize_end - initialize_start)

    while (1):
        mcmc_start = timer()
        for t in range(1,tmax+1):
            if (conf. SAMPLEPOINTS_DEBUGGER == conf.ON):
                if (t % 1000 == 0): 
                    prettyprint_samplepoints(samplepoints, "Samplepoints Now", "\t") 

            if (costI == 0): #Put this in the start, because if by some magic we guess the first invariant in the first go, we dont want to change
                break  
            
            index = get_index(repr.get_d(), repr.get_c())
            oldpredicate = I_guess[index[0]][index[1]]
            is_rotationchange = isrotationchange()
            if (is_rotationchange ): 
                rotneighbors = repr.get_coeffneighbors(oldpredicate[:-2])
                deg = rotationdegree(rotneighbors)
                # Get required points from samplepoints and costlist
                filteredpoints = getrotationcentre_points(samplepoints, costlist, oldpredicate) 
                newpred = rotationtransition(oldpredicate, rotneighbors, spinI, repr.get_k1(), filteredpoints) 
                degnew = rotationdegree(repr.get_coeffneighbors(newpred[:-2]))
                I_guess[index[0]][index[1]] = newpred
            else:
                newpred = translationtransition(oldpredicate) 
                I_guess[index[0]][index[1]] = newpred
                (deg, degnew) = (2,2)
                
            LII = dnfconjunction( list3D_to_listof2Darrays(I_guess), repr.get_affineSubspace(), 0)
            (costInew, costlist, spinInew) = cost(LII, samplepoints)
            temp = conf.temp_C/log(1 + t)
            fInew = cost_to_f(costInew, temp)
            if (costInew <= costI):
                a = 1.0
            else:
                if (fI == 0 or isnan(fI)):
                    if (fInew == 0 or isnan(fInew)): #Handle underflow!
                        a = 0.5                    
                    else:
                        a = 1.0
                else:
                    a = min( ( fInew * deg)/ (fI * degnew) , 1.0) 
            if (random.rand() <=  a): 
                reject = 0
                descent = 1 if (costInew > costI) else 0 
                (fI, costI, spinI) = (fInew, costInew, spinInew)
                statistics(t, I_guess, fInew, costInew, descent, reject, costlist,a , repr.get_Var() )                   
            else:
                reject = 1
                descent = 0
                statistics(t, I_guess, fInew, costInew, descent, reject, costlist, a, repr.get_Var()  )
                I_guess[index[0]][index[1]] = oldpredicate
            
        mcmc_end = timer()
        mcmc_time = mcmc_time + (mcmc_end - mcmc_start)
        total_iterations = total_iterations + t
        
        z3_start = timer()
        LII = dnfconjunction( list3D_to_listof2Darrays(I_guess), repr.get_affineSubspace(), 0)
        (z3_correct, cex) = z3_verifier(repr.get_P_z3expr(), repr.get_B_z3expr(), repr.get_T_z3expr(), repr.get_Q_z3expr(), LII )           
        z3_end = timer()
        z3_time = z3_time + (z3_end - z3_start)

        initialize_start = timer()
        z3_callcount = z3_callcount + 1
        z3statistics(z3_correct, samplepoints, cex, z3_callcount, (t == tmax))
        if (z3_correct):
            break        
        elif ((not z3_correct) and (t == tmax)):
            return ("No Invariant Found", "-", z3_callcount)
        samplepoints = (samplepoints[0] + cex[0] , samplepoints[1] + cex[1], samplepoints[2] + cex[2])
        (costI, costlist, spinI) = cost( LII, samplepoints) #samplepoints has changed, so cost and f changes for same invariant
        temp = conf.temp_C/log(2)
        fI = cost_to_f(costI, temp)
        statistics(0, I_guess, fI, costI, 0, 0, [], -1 , repr.get_Var() )
        initialize_end = timer()
        initialize_time = initialize_time + (initialize_end - initialize_start)

    invariantfound(repr.get_nonItersP(), repr.get_affineSubspace(), I_guess, repr.get_Var())
    timestatistics(mcmc_time, total_iterations, z3_time, initialize_time, z3_callcount )

    return (LII, fI, z3_callcount)



