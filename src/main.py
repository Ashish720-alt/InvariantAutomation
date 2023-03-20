""" Imports.
"""
from configure import Configure as conf
from cost_funcs import cost, cost_to_f
from guess import uniformlysample_I, rotationdegree, rotationtransition, translationtransition, get_index, isrotationchange, getrotationcentre_points
from repr import Repr
from numpy import random
from z3verifier import z3_verifier
from print import initialized, statistics, z3statistics, invariantfound, timestatistics, prettyprint_samplepoints, noInvariantFound
import copy
from dnfs_and_transitions import RTI_to_LII, list3D_to_listof2Darrays, dnfconjunction
from timeit import default_timer as timer
from math import isnan, log
import argparse
from input import Inputs, input_to_repr
import multiprocessing as mp
# from numba import jit


# @jit(nopython=False)
def search(repr: Repr, I, samplepoints, process_id, return_value):
    tmax = repr.get_tmax()
    mcmc_start = timer()
    LII = dnfconjunction(list3D_to_listof2Darrays(I), repr.get_affineSubspace() , 0)
    (costI, costlist, spinI) = cost(LII, samplepoints)  #spin = |-| - |+|
    temp = conf.temp_C/log(2)
    fI = cost_to_f(costI, temp)


    for t in range(1, tmax+1):
        if (conf.SAMPLEPOINTS_DEBUGGER == conf.ON):
            if (t % 1000 == 0):
                for i in range(conf.num_processes):
                    if return_value[i] != None:
                        print("Process ", process_id, " exit early")
                        return
                prettyprint_samplepoints(samplepoints,
                                         "Samplepoints Now", "\t")
        if (costI == 0):
            return_value[process_id] = (I, (0, 0, 0))
            print("Process ", process_id, " found invariant")
            return
        index = get_index(repr.get_d(), repr.get_c())
        oldpredicate = I[index[0]][index[1]]
        is_rotationchange = isrotationchange()
        if (is_rotationchange):
            rotneighbors = repr.get_coeffneighbors(oldpredicate[:-2])
            deg = rotationdegree(rotneighbors)
            # Get required points from samplepoints and costlist
            filteredpoints = getrotationcentre_points(samplepoints, costlist, oldpredicate)
            newpred = rotationtransition(oldpredicate, rotneighbors, spinI, repr.get_k1(), filteredpoints) 
            degnew = rotationdegree(repr.get_coeffneighbors(newpred[:-2]))
            I[index[0]][index[1]] = newpred
        else:
            newpred = translationtransition(oldpredicate) 
            I[index[0]][index[1]] = newpred
            (deg, degnew) = (conf.translation_range,conf.translation_range)
        
        LII = dnfconjunction( list3D_to_listof2Darrays(I), repr.get_affineSubspace(), 0)
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
                a = min((fInew * deg) / (fI * degnew), 1.0) 
        if (random.rand() <= a): 
            reject = 0
            descent = 1 if (costInew > costI) else 0
            (fI, costI, spinI) = (fInew, costInew, spinInew)
        else:
            reject = 1
            descent = 0
            I[index[0]][index[1]] = oldpredicate
        print("Process: ", process_id, " Iteration: ", t)
        statistics(t, I, fInew, costInew, descent, reject, costlist, a, repr.get_Var())

    mcmc_end = timer()
    mcmc_time = mcmc_time + (mcmc_end - mcmc_start)
    total_iterations = total_iterations + t

    # Failed
    print("Process ", process_id, " failed")
    return


""" Main function. """


def main(repr: Repr):
    """ ===== Initialization starts. ===== """
    mcmc_time = 0
    total_iterations = 0
    z3_time = 0
    initialize_time = 0
    z3_callcount = 0

    initialize_start = timer()
    tmax = repr.get_tmax()
    samplepoints = (repr.get_plus0(), repr.get_minus0(), repr.get_ICE0())
    initialized( repr.get_affineSubspace(), repr.get_nonItersP(), repr.get_Var())
    prettyprint_samplepoints(samplepoints, "Selected-Points", "\t")
    print("\n")

    I_list = []
    for i in range(conf.num_processes):
        I_guess = uniformlysample_I( repr.get_coeffvertices(), repr.get_k1(), repr.get_c(), repr.get_d(), repr.get_n())
        LII = dnfconjunction( list3D_to_listof2Darrays(I_guess), repr.get_affineSubspace() , 0)
        (costI, costlist, spinI) = cost(LII, samplepoints)  #spin = |-| - |+|
        temp = conf.temp_C/log(2)
        fI = cost_to_f(costI, temp)
        statistics(0, I_guess, fI, costI, 0, 0, costlist, -1, repr.get_Var() ) 
        I_list.append(I_guess)
    initialize_end = timer()
    initialize_time = initialize_time + (initialize_end - initialize_start)
    """ ===== Initialization ends. ===== """

    while (1):
        """ Searching Loop """
        process_list = []
        manager = mp.Manager()
        return_value = manager.list()
        return_value.extend([None for i in range(conf.num_processes)])
        for i in range(conf.num_processes):
            process_list.append(mp.Process(target = search, args = (repr, I_list[i], samplepoints, i, return_value)))
            process_list[i].start()
        
        for i in range(conf.num_processes):
            process_list[i].join()

        for result in return_value:
            print(result)
            if (result != None):
                (I, (t, mcmc_time, total_iterations)) = result
                break

        """ Z3 validation """
        z3_callcount = z3_callcount + 1
        LII = dnfconjunction( list3D_to_listof2Darrays(I), repr.get_affineSubspace(), 0)
        z3_start = timer()
        (z3_correct, cex) = z3_verifier(repr.get_P_z3expr(), repr.get_B_z3expr(), repr.get_T_z3expr(), repr.get_Q_z3expr(), LII )           
        z3_end = timer()
        z3_time = z3_time + (z3_end - z3_start)
        z3statistics(z3_correct, samplepoints, cex, z3_callcount, (t == tmax))

        """ Collect counterexamples """
        initialize_start = timer()
        if (z3_correct):
            break      
        elif ((not z3_correct) and (t == tmax)):
            noInvariantFound(z3_callcount)
            return ("No Invariant Found", "-", z3_callcount)
        samplepoints = (samplepoints[0] + cex[0] , samplepoints[1] + cex[1], samplepoints[2] + cex[2])
        (costI, costlist, spinI) = cost(LII, samplepoints) #samplepoints has changed, so cost and f changes for same invariant
        temp = conf.temp_C/log(2)
        fI = cost_to_f(costI, temp)
        statistics(0, I, fI, costI, 0, 0, [], -1 , repr.get_Var())
        initialize_end = timer()
        initialize_time = initialize_time + (initialize_end - initialize_start)

    invariantfound(repr.get_nonItersP(), repr.get_affineSubspace(), I, repr.get_Var())
    timestatistics(mcmc_time, total_iterations, z3_time, initialize_time, z3_callcount )

    return (LII, fI, z3_callcount)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MCMC Invariant Search')
    parser.add_argument('-c', type=int, help='Number of conjunctions')
    parser.add_argument('-d', type=int, help='Number of disjunctions')
    parser.add_argument('-i', '--input', type=str, help='Input object name')
    parser.add_argument('-a', '--all', action='store_true', help='Run all inputs')
    parse_res = vars(parser.parse_args())
    if parse_res['all']:
        if (parse_res['input'] is not None):
            print(parser.print_help())
            print("Please specify either input object name or all inputs")
            exit(1)
        for subfolder in dir(Inputs):
            for input in dir(getattr(Inputs, subfolder)):
                print(main(input_to_repr(getattr(getattr(Inputs, subfolder), input), parse_res['c'], parse_res['d'])))
    else:
        if parse_res['input'] is None:
            print(parser.print_help())
            print("Please specify input object name")
            exit(1)
        else:
            (first_name, last_name) = parse_res['input'].split('.')

            for subfolder in Inputs.__dict__:
                if subfolder == first_name:
                    for input in getattr(Inputs, subfolder).__dict__:
                        if input == last_name:
                            print(main(input_to_repr(getattr(getattr(Inputs, subfolder), input), parse_res['c'], parse_res['d'])))
