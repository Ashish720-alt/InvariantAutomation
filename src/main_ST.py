""" Imports.
"""
import sys
from configure import Configure as conf
from cost_funcs import cost
from guess import initialInvariant, rotationtransition, translationtransition, get_index, isrotationchange, k1list, SAconstantlist, getNewRotConstant, getNewTranslationConstant, experimentalSAconstantlist
from repr import Repr
from numpy import random
from z3verifier import z3_verifier
from print import initialized, statistics, z3statistics, invariantfound, timestatistics, prettyprint_samplepoints, noInvariantFound
from print import SAexit, SAsuccess, samplepoints_debugger, SAfail
from dnfs_and_transitions import  list3D_to_listof2Darrays, dnfconjunction, dnfnegation
from timeit import default_timer as timer
from math import log, floor, exp, e
import argparse
from input import Inputs, input_to_repr
import multiprocessing as mp
from invariantspaceplotter import plotinvariantspace
from selection_points import removeduplicates, removeduplicatesICEpair, get_longICEpairs
from costplotter import CostPlotter
import stagnation


def ST_cost(cost, E_0):
    ST_gamma = 1.0  #Vary this hyperparameter
    p = exp(- ST_gamma * (cost - E_0))
    return 1 - p


# @jit(nopython=False)
def search(repr: Repr, I_list, samplepoints, process_id, return_value, SA_Gamma, z3_callcount, k1, costTimeLists):
    
    
    #Important for truly random processes (threads).
    random.seed()
    
    I = I_list[process_id]
    n = repr.get_n()
    tmax = repr.get_tmax()
    LII = dnfconjunction(list3D_to_listof2Darrays(I_list[process_id]), repr.get_affineSubspace() , 0)
    (costI, costlist) = cost(LII, samplepoints)  


    E_0 = costI


    if (conf.CHECK_STAGNATION == conf.ON):
        stagnant = False
        costTimeList = [costI]

    for t in range(1, tmax+1):
        if (t % conf.NUM_ROUND_CHECK_EARLY_EXIT == 0):
            for i in range(conf.num_processes):
                if return_value[i] != None and return_value[i][0] != None:
                    return_value[process_id] = (None, t)
                    SAexit(process_id, repr.get_colorslist())
                    I_list[process_id] = I
                    return

        samplepoints_debugger(repr.get_n(), process_id, z3_callcount, t, samplepoints, I, repr.get_colorslist())        

        
        if (conf.COST_PLOTTER == conf.ON):
            # tmp = costTimeLists[process_id]
            # tmp.append(costI)  
            # costTimeLists[process_id] = tmp
            costTimeLists[process_id] = costTimeLists[process_id] + [costI]
           
        if (costI == 0):
            return_value[process_id] = (I, t)
            SAsuccess(process_id, repr.get_colorslist())
            I_list[process_id] = I
            return
        
        neighbors = []
        for i in range(repr.get_d()):
            for j in range(repr.get_c()):
                oldcoeff = I[i][j][:-2]
                oldconst = I[i][j][-1]
                if (n <= 3):
                    rotneighbors = repr.get_coeffneighbors(oldcoeff)
                else:
                    negative_indices = [idx for idx, val in enumerate(oldcoeff) if val < 0]
                    rotneighbors = [ [-coeff[i] if i in negative_indices else coeff[i] for i in range(n)]  
                                            for coeff in repr.get_coeffneighbors([abs(val) for val in oldcoeff])]
                for r in rotneighbors:
                    constlist = getNewRotConstant(oldcoeff, oldconst, r, k1)
                    for c in constlist:
                        neighbors.append( ( i, j, r + [-1,c]) )
                transconslist = getNewTranslationConstant(oldcoeff, oldconst, k1)
                for c in transconslist:
                    neighbors.append( ( i, j, oldcoeff + [-1,c]) )
        
        # if (conf.CHECK_STAGNATION == conf.ON and conf.CHECK_LOCALMINIMA == conf.ON):
        #     if (stagnant):
        #         if (stagnation.checkLocalMinima(I, neighbors, samplepoints)):
        #             print(repr.get_colorslist()[process_id] + "Process " + str(process_id) + " is stuck in a Local Minima!")
        #         else:
        #             print(repr.get_colorslist()[process_id] + "Process " + str(process_id) + " is NOT stuck in a Local Minima.")
        
        deg = len(neighbors)
        P = neighbors[random.choice(range(deg))]
        oldpredicate = I[P[0]][P[1]] 
        I[P[0]][P[1]] = P[2]
                    
        # index = get_index(repr.get_d(), repr.get_c())
        
        # oldpredicate = I[index[0]][index[1]]
        # rotneighbors = repr.get_coeffneighbors(oldpredicate[:-2])
        
        # if (isrotationchange(oldpredicate, rotneighbors, k1)):
        #     I[index[0]][index[1]] = rotationtransition(oldpredicate, rotneighbors, k1) 
        # else:
        #     I[index[0]][index[1]] = translationtransition(oldpredicate, k1) 
        
        
        LII = dnfconjunction( list3D_to_listof2Darrays(I), repr.get_affineSubspace(), 0)
        (costInew, costlist) = cost(LII, samplepoints)

        if (costInew < E_0):
            E_0 = costInew
        
        beta = 1
        # print(costI, ST_cost(costI, E_0)) #Check transformed values!
        a = conf.e **( beta * -max( ST_cost(costInew, E_0) - ST_cost(costI, E_0), 0.0) ) 
        if (random.rand() <= a): 
            reject = 0
            descent = 1 if (costInew > costI) else 0
            costI = costInew
            statistics(process_id, t, I, costInew, descent, reject, costlist, a, repr.get_Var(), repr.get_colorslist())
        else:
            reject = 1
            descent = 0
            statistics(process_id, t, I, costInew, descent, reject, costlist, a, repr.get_Var(), repr.get_colorslist()) #Print rejected value
            I[P[0]][P[1]] = oldpredicate
            
        #Local Minima Checker!!
        if (conf.CHECK_STAGNATION == conf.ON):
            if len(costTimeList) >= conf.STAGNANT_TIME_WINDOW:
                costTimeList = costTimeList[1:]      
            costTimeList = costTimeList + [costI]
            stagnant = stagnation.isStagnant(costTimeList)
            if (stagnant):
                if (conf.CHECK_LOCALMINIMA == conf.ON):
                    print(repr.get_colorslist()[process_id] + "Process " + str(process_id) + " has stagnated!")
                    localMinimastring = "" if stagnation.checkLocalMinima(I, repr, samplepoints) else "NOT"
                    print(repr.get_colorslist()[process_id] + "Process " + str(process_id) + " is " + localMinimastring + " stuck in a Local Minima.")  
                
                #Printing localArea not feasible for even 2 neighbors (~17,000 elements)
                # localAreaCosts = stagnation.checkAreaAroundStuck(I, repr, samplepoints)
                # for i in range(1,conf.STAGNATION_AREA_CHECK + 1):
                #     print(repr.get_colorslist()[process_id] + sorted(localAreaCosts[i])[:5], '\n', localAreaCosts[i])       
                
                stagnation.gradientdescent(I, repr, samplepoints, repr.get_colorslist()[process_id])
        
        # statistics(process_id, t, I, costInew, descent, reject, costlist, a, repr.get_Var(), repr.get_colorslist())

    # Process 'process_id' Failed!
    SAfail(process_id, repr.get_colorslist())
    return_value[process_id] = (None, t)
    I_list[process_id] = I
    return 


""" Main function. """


def main(inputname, repr: Repr):
    """ ===== Initialization starts. ===== """
    mcmc_time = 0
    z3_time = 0
    initialize_time = 0
    z3_callcount = 0
    initialize_start = timer()
    tmax = repr.get_tmax()
    samplepoints = (repr.get_plus0(), repr.get_minus0(), repr.get_ICE0())
    initialized( repr.get_affineSubspace(), repr.get_nonItersP(), repr.get_Var())
    prettyprint_samplepoints(samplepoints, "Selected-Points", "\t")
    print("\n")

    if (conf.INVARIANTSPACE_PLOTTER == conf.ON):
        plotinvariantspace(3, repr.get_coeffedges(), samplepoints, repr.get_c(), repr.get_d(), 0)

    manager = mp.Manager()
    I_list = manager.list()
    
    
    (I_guess, _) = initialInvariant(samplepoints, repr.get_coeffvertices(), repr.get_k1(), repr.get_c(), repr.get_d(), repr.get_n(), 
                                    repr.get_affineSubspace(), repr.get_Dp())
    
    
    LII = dnfconjunction( list3D_to_listof2Darrays(I_guess), repr.get_affineSubspace() , 0)
    (costI, costlist) = cost(LII, samplepoints)  
    for i in range(conf.num_processes):
        statistics(i, 0, I_guess, costI, 0, 0, costlist, -1, repr.get_Var(), repr.get_colorslist() ) 
        I_list.append(I_guess.copy())
    initialize_end = timer()
    initialize_time = initialize_time + (initialize_end - initialize_start)
    """ ===== Initialization ends. ===== """

    mcmc_time = 0.0
    mcmc_iterations = 0
    while (1):
        """ Searching Loop """
        process_list = []
        return_value = manager.list()
        return_value.extend([None for i in range(conf.num_processes)])
        
        mcmc_start = timer()
        
        k1_list = k1list(repr.get_k0(), repr.get_n())
        SA_gammalist = experimentalSAconstantlist()       
        
        costTimeLists = manager.list()
        costTimeLists.extend([[] for i in range(conf.num_processes)])
        # costTimeLists = {}
        for i in range(conf.num_processes):
            
            process_list.append(mp.Process(target = search, args = (repr, I_list, samplepoints, i, return_value,
                                                                    SA_gammalist[i], z3_callcount, k1_list[i], costTimeLists )))
            process_list[i].start()
        
        for i in range(conf.num_processes):
            process_list[i].join()

        if (conf.COST_PLOTTER == conf.ON):
            CostPlotter( costTimeLists , conf.num_processes, filename = inputname + '_Z3Calls' + str(z3_callcount) + ".png" )

        mcmc_end = timer()
        mcmc_time = mcmc_time + (mcmc_end - mcmc_start)     
            
        """ Z3 validation """
        z3_start = timer()
        
        z3_callcount = z3_callcount + 1
        foundI = False
        cex = ([], [], [])
        for result in return_value:
            (I, t) = result    
            foundI = foundI or (I != None)
            mcmc_iterations = mcmc_iterations + t + 1            
            if (I != None):
                LII = dnfconjunction( list3D_to_listof2Darrays(I), repr.get_affineSubspace(), 0)
                (z3_correct, cex_thread) = z3_verifier(repr.get_P_z3expr(), repr.get_B_z3expr(), repr.get_T_z3expr(), repr.get_Q_z3expr(), LII )           
                if (z3_correct):
                    break
                cex = (removeduplicates(cex[0] + cex_thread[0]) , removeduplicates(cex[1] + cex_thread[1]) , removeduplicatesICEpair(cex[2] + cex_thread[2]) )

        # Print the same thing again to the end of "output/{inputname}.txt"
        #     with open("output/" + inputname + ".txt", "a") as f:
        #         ori_stdout = sys.stdout
        #         sys.stdout = f
        #         noInvariantFound(z3_callcount) 
        #         print("-------------------\n")
        #         sys.stdout = ori_stdout
        if (z3_correct):
            z3_end = timer()
            z3_time = z3_time + (z3_end - z3_start)
            break   
        elif foundI == False:
            noInvariantFound(z3_callcount)
            z3_end = timer()
            z3_time = z3_time + (z3_end - z3_start)
            return ("All thread time out", z3_callcount)
        
        new_enet = (z3_callcount % conf.z3_stepwindow == 0)
        i = floor(1.0 * z3_callcount / conf.z3_stepwindow)
        e = conf.e0 / (2**i)
        
        # Constricting e-net
        eNetPoints = ([], [], [])
        if (new_enet):
            eNetPoints = repr.update_enet(e, samplepoints)

        z3_end = timer()
        z3_time = z3_time + (z3_end - z3_start)

        #Get iterated implication pairs 
        iteratedImplicationpairs = get_longICEpairs( cex[2], repr.get_T(), repr.get_n(), repr.get_transitionIterates())
        
        z3statistics(z3_correct, samplepoints, cex, z3_callcount, (t == tmax), new_enet, e , eNetPoints, iteratedImplicationpairs)

        """ Collect counterexamples """
        initialize_start = timer()
        
        
        cex = (cex[0] , cex[1] , cex[2] + iteratedImplicationpairs )
        
        if (new_enet):
            samplepoints = ( removeduplicates( samplepoints[0] + eNetPoints[0] + cex[0]), 
                            removeduplicates( samplepoints[1] + eNetPoints[1] + cex[1] ),
                            removeduplicatesICEpair( samplepoints[2] + eNetPoints[2] + cex[2]) )
        else:
            samplepoints = (removeduplicates(samplepoints[0] + cex[0]) , 
                            removeduplicates(samplepoints[1] + cex[1]), 
                            removeduplicatesICEpair(samplepoints[2] + cex[2]) )

        if (conf.INVARIANTSPACE_PLOTTER == conf.ON):
            plotinvariantspace(5, repr.get_coeffedges(), samplepoints, repr.get_c(), repr.get_d(), z3_callcount)
        
        #samplepoints has changed, so cost and f changes for same invariant
        for i in range(conf.num_processes):
            LII = dnfconjunction( list3D_to_listof2Darrays(I_list[i]), repr.get_affineSubspace(), 0)
            (costI, costlist) = cost(LII, samplepoints)
            statistics(i, 0, I_list[i], costI, 0, 0, [], -1 , repr.get_Var(), repr.get_colorslist()) 
        
        
        initialize_end = timer()
        initialize_time = initialize_time + (initialize_end - initialize_start)

    invariantfound(repr.get_nonItersP(), repr.get_affineSubspace(), I, repr.get_Var())
    timestatistics(mcmc_time, mcmc_iterations, z3_time, initialize_time, z3_callcount, conf.num_processes )

    # Print the same thing again to the end of "output/{inputname}.txt"
    # with open("output/" + inputname + ".txt", "a") as f:
    #     ori_stdout = sys.stdout
    #     sys.stdout = f
    #     invariantfound(repr.get_nonItersP(), repr.get_affineSubspace(), I, repr.get_Var())
    #     timestatistics(mcmc_time, mcmc_iterations, z3_time, initialize_time, z3_callcount, conf.num_processes )
    #     print("-------------------\n")
    #     sys.stdout = ori_stdout
        

    return (LII, z3_callcount)


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
            for inp in dir(getattr(Inputs, subfolder)):
                main(input_to_repr(getattr(getattr(Inputs, subfolder), inp), parse_res['c'], parse_res['d']))
    else:
        if parse_res['input'] is None:
            print(parser.print_help())
            print("Please specify input object name")
            exit(1)
        else:
            (first_name, last_name) = parse_res['input'].split('.')
            for subfolder in Inputs.__dict__:
                if subfolder == first_name:
                    for inp in getattr(Inputs, subfolder).__dict__:
                        if inp == last_name:
                            main(first_name + "." + last_name, input_to_repr(getattr(getattr(Inputs, subfolder), inp), parse_res['c'], parse_res['d']))
