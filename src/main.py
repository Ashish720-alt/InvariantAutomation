""" Imports.
"""
import sys
from configure import Configure as conf
from cost_funcs import cost
from guess import initialInvariant,  experimentalSAconstantlist, SearchSpaceNeighbors, FasterBiased_RWcostlist
from repr import Repr
from numpy import random
from z3verifier import z3_verifier
from print import initialized, statistics, z3statistics, invariantfound, timestatistics, prettyprint_samplepoints, noInvariantFound
from print import SAexit, SAsuccess, n2plotter, SAfail, print_with_mode, list_to_string, printTemperaturePrompt
from dnfs_and_transitions import  list3D_to_listof2Darrays, dnfconjunction, dnfnegation
from timeit import default_timer as timer
from math import log, floor, e
import argparse
from input import Inputs, input_to_repr
import multiprocessing as mp
from invariantspaceplotter import plotinvariantspace
from selection_points import removeduplicates, removeduplicatesICEpair, get_longICEpairs
from costplotter import CostPlotter
from stagnation import checkLocalMinima, isStagnant
from colorama import Fore
from gradientDescent import gradientDescent
from EvolutionaryAlgorithm import geneticProgramming

# @jit(nopython=False)
def simulatedAnnealing(inputname, repr: Repr, I_list, samplepoints, process_id, return_value, SA_Gamma, z3_callcount, costTimeLists, output ):
    #Important for truly random processes (threads).
    random.seed()
    
    I = I_list[process_id]
    n = repr.get_n()
    tmax = repr.get_tmax()
    LII = dnfconjunction(list3D_to_listof2Darrays(I_list[process_id]), repr.get_affineSubspace() , 0)
    (costI, costlist) = cost(LII, samplepoints)  

    temp = SA_Gamma /log(2)


    if (conf.CHECK_LOCALMINIMA  == conf.ON):
        costSingleTimeList = [costI]
    if (conf.AVERAGE_ACC_PROB_CHECKER == conf.ON):
        aList = []

    for t in range(1, tmax+1):
        if (t % conf.NUM_ROUND_CHECK_EARLY_EXIT == 0):
            for i in range(conf.num_processes):
                if return_value[i] != None and return_value[i][0] != None:
                    return_value[process_id] = (None, t)
                    SAexit(process_id, repr.get_colorslist(), output)
                    I_list[process_id] = I
                    return

        if (conf.n2PLOTTER == conf.ON):
            n2plotter(inputname.split('.')[1], repr.get_n(), process_id, z3_callcount, t, samplepoints, I, repr.get_colorslist(), outputfile = output)        
        if (conf.COST_PLOTTER == conf.ON):
            costTimeLists[process_id] = costTimeLists[process_id] + [costI]
           
        if (costI == 0):
            return_value[process_id] = (I, t)
            SAsuccess(process_id, repr.get_colorslist(), output)
            I_list[process_id] = I
            return
        
        neighbors = SearchSpaceNeighbors(I, repr, repr.get_d(), repr.get_cList(), repr.get_k1(), n) 
        deg = len(neighbors)
        P = neighbors[random.choice(range(deg))]
        oldpredicate = I[P[0]][P[1]] 
        I[P[0]][P[1]] = P[2]
        LII = dnfconjunction( list3D_to_listof2Darrays(I), repr.get_affineSubspace(), 0)
        (costInew, costlist) = cost(LII, samplepoints)
        temp = SA_Gamma/log(conf.t0 + t)
        a = e **( - max(costInew - costI, 0.0) / temp ) 
        if (conf.AVERAGE_ACC_PROB_CHECKER == conf.ON and a < 1):
            aList = aList + [a]    
        if (random.rand() <= a): 
            reject = 0
            descent = 1 if (costInew > costI) else 0
            costI = costInew
            statistics(process_id, t, I, costInew, descent, reject, costlist, a, repr.get_Var(), repr.get_colorslist(), output)
        else:
            reject = 1
            descent = 0
            statistics(process_id, t, I, costInew, descent, reject, costlist, a, repr.get_Var(), repr.get_colorslist(), output) #Print rejected value
            I[P[0]][P[1]] = oldpredicate
            
        if (conf.CHECK_LOCALMINIMA  == conf.ON):
            if len(costSingleTimeList) >= conf.STAGNANT_TIME_WINDOW:
                costSingleTimeList = costSingleTimeList[1:]      
            costSingleTimeList = costSingleTimeList + [costI]
            stagnant = isStagnant(costSingleTimeList) #Heuristic Local Minima Checker
            if (stagnant):
                print_with_mode(repr.get_colorslist()[process_id], "Process " + str(process_id) + " has stagnated!", endstr= '\n', file= output)
                checkLocalMinima(I, repr, samplepoints, repr.get_colorslist()[process_id]) #Heuristic Algorithm to find how far from Local Minima
        if (conf.AVERAGE_ACC_PROB_CHECKER == conf.ON and t % conf.AVERAGE_ACC_PROB_WINDOW == 0):
            if (len(aList) < 0):
                print_with_mode(repr.get_colorslist()[process_id], "Process " + str(process_id) + " for window " + 
                                    str(t // conf.AVERAGE_ACC_PROB_WINDOW) + ' made no positive transitions.', endstr = '\n', file = output)
            else:
                print_with_mode(repr.get_colorslist()[process_id], "Average a for Process " + str(process_id) + " for window " + 
                                    str(t // conf.AVERAGE_ACC_PROB_WINDOW) + ' is ' + str( sum(aList)/ len(aList)), endstr = '\n', file = output)
            aList = []

    # Process 'process_id' Failed!
    SAfail(process_id, repr.get_colorslist(), output)
    return_value[process_id] = (None, t)
    I_list[process_id] = I
    return 


""" Main function. """
def main(inputname, repr: Repr):
    """ ===== Initialization starts. ===== """

    if(conf.PRINTING_MODE == conf.FILE or conf.PRINTING_MODE == conf.TERMINAL_AND_FILE):
        outputF = open("output/" + inputname + ".txt", 'w')
    else:
        outputF = None
    
    mcmc_time = 0
    z3_time = 0
    initialize_time = 0
    z3_callcount = 0
    initialize_start = timer()
    
    tmax = repr.get_tmax()
    samplepoints = (repr.get_plus0(), repr.get_minus0(), repr.get_ICE0())
    initialized( repr.get_affineSubspace(), repr.get_nonItersP(), repr.get_Var(), outputfile = outputF)
    if (conf.PRINT_Z3_ITERATIONS == conf.ON):
        prettyprint_samplepoints(samplepoints, "Selected-Points", "\t", outputfile = outputF)

    if (conf.INVARIANTSPACE_PLOTTER == conf.ON):
        plotinvariantspace(conf.INVARIANTSPACE_MAXCONST, repr.get_coeffedges(), samplepoints, repr.get_c(), repr.get_d(), 0)

    manager = mp.Manager()
    I_list = manager.list()
    
    
    (I_guess, _) = initialInvariant(samplepoints, repr.get_coeffvertices(), repr.get_k1(), repr.get_cList(), repr.get_d(), repr.get_n(), 
                                    repr.get_affineSubspace(), repr.get_Dp())
    LII = dnfconjunction( list3D_to_listof2Darrays(I_guess), repr.get_affineSubspace() , 0)
    (costI, costlist) = cost(LII, samplepoints)  
    for i in range(conf.num_processes):
        statistics(i, 0, I_guess, costI, 0, 0, costlist, -1, repr.get_Var(), repr.get_colorslist(), outputF ) 
        I_list.append(I_guess.copy())
    initialize_end = timer()
    initialize_time = initialize_time + (initialize_end - initialize_start)
    """ ===== Initialization ends. ===== """

    mcmc_time = 0.0
    mcmc_iterations = 0
    while (1):
        """ Search Based Algorithm (eg: simulated annealing) Loop """
        process_list = []
        return_value = manager.list()
        return_value.extend([None for i in range(conf.num_processes)])
        
        mcmc_start = timer()
        
        if (costI != 0):
            printTemperaturePrompt(repr.get_colorslist(), outputF)
            SA_gammalist = experimentalSAconstantlist( I_list, samplepoints, repr  ) 
            if (conf.PRINT_ITERATIONS == conf.ON):
                print_with_mode(Fore.WHITE, "T0 values are " + list_to_string(SA_gammalist) ,endstr = '\n', file = outputF)
            
            costTimeLists = manager.list()
            costTimeLists.extend([[] for i in range(conf.num_processes)])
            for i in range(conf.num_processes):
                if (conf.SEARCH_STRATEGY == conf.SA):
                    process_list.append(mp.Process(target = simulatedAnnealing, args = (inputname, repr, I_list, samplepoints, i, return_value,
                                                                        SA_gammalist[i], z3_callcount, costTimeLists, outputF )))
                elif (conf.SEARCH_STRATEGY == conf.GD):
                    process_list.append(mp.Process(target = gradientDescent , args = (inputname, repr, I_list, samplepoints, i, return_value,
                                                                        _ , z3_callcount, costTimeLists, outputF )))                    
                elif (conf.SEARCH_STRATEGY == conf.EA):
                    process_list.append(mp.Process(target = geneticProgramming , args = (inputname, repr, I_list, samplepoints, i, return_value,
                                                                        _ , z3_callcount, costTimeLists, outputF ))) 
                process_list[i].start()
                
            for i in range(conf.num_processes):
                process_list[i].join()

            if (conf.COST_PLOTTER == conf.ON):
                CostPlotter( costTimeLists , conf.num_processes, filename = 'CostPlots/' + inputname + '_Z3Calls' + str(z3_callcount) + ".png" )
        else:
            for i in range(conf.num_processes):
                return_value[i] = (I_list[i], 1)

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

        if (z3_correct):
            z3_end = timer()
            z3_time = z3_time + (z3_end - z3_start)
            break   
        elif foundI == False:
            noInvariantFound(z3_callcount, outputfile = outputF)
            z3_end = timer()
            z3_time = z3_time + (z3_end - z3_start)        
            if(conf.PRINTING_MODE == conf.FILE or conf.PRINTING_MODE == conf.TERMINAL_AND_FILE):
                outputF.close()
            return ("All threads time out", z3_callcount)
        
        new_enet = (z3_callcount % conf.z3_stepwindow == 0)
        i = floor(1.0 * z3_callcount / conf.z3_stepwindow)
        e = conf.e0 / (2**i)
        
        # Constricting e-net
        eNetPoints = ([], [], [])
        if (new_enet):
            eNetPoints = repr.update_enet(e, samplepoints)

        #Get iterated implication pairs 
        iteratedImplicationpairs = get_longICEpairs( cex[2], repr.get_T(), repr.get_n(), repr.get_transitionIterates())
        z3statistics(z3_correct, samplepoints, cex, z3_callcount, (t == tmax), new_enet, e , eNetPoints, iteratedImplicationpairs, outputF)
        z3_end = timer()
        z3_time = z3_time + (z3_end - z3_start)
        
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
            plotinvariantspace(conf.INVARIANTSPACE_MAXCONST, repr.get_coeffedges(), samplepoints, repr.get_c(), repr.get_d(), z3_callcount)
        
        #For n = 1, random sampling again gives better results.
        if (repr.get_n() == 1):
            (I_guess, _) = initialInvariant(samplepoints, repr.get_coeffvertices(), repr.get_k1(), repr.get_cList(), repr.get_d(), repr.get_n(), 
                                    repr.get_affineSubspace(), repr.get_Dp())        
            for i in range(conf.num_processes):
                # statistics(i, 0, I_guess, costI, 0, 0, costlist, -1, repr.get_Var(), repr.get_colorslist(), outputF ) 
                I_list.append(I_guess.copy())        
        #samplepoints has changed, so cost and f changes for same invariant
        for i in range(conf.num_processes):
            LII = dnfconjunction( list3D_to_listof2Darrays(I_list[i]), repr.get_affineSubspace(), 0)
            (costI, costlist) = cost(LII, samplepoints)
            statistics(i, 0, I_list[i], costI, 0, 0, [], -1 , repr.get_Var(), repr.get_colorslist(), outputF) 
        
        
        initialize_end = timer()
        initialize_time = initialize_time + (initialize_end - initialize_start)

    invariantfound(repr.get_nonItersP(), repr.get_affineSubspace(), I, repr.get_Var(), outputF)
    timestatistics(mcmc_time, mcmc_iterations, z3_time, initialize_time, z3_callcount, conf.num_processes, outputfile = outputF )
    if(conf.PRINTING_MODE == conf.FILE or conf.PRINTING_MODE == conf.TERMINAL_AND_FILE):
        outputF.close()
    
    
    return (LII, z3_callcount)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SA-CEGUS Invariant Search')
    parser.add_argument('-c', type=int, help='Number of conjunctions')
    parser.add_argument('-d', type=int, help='Number of disjunctions')
    parser.add_argument('-clist', type=list, help='List of c values')
    parser.add_argument('-i', '--input', type=str, help='Input object name')
    # parser.add_argument('-a', '--all', action='store_true', help='Run all inputs')
    parse_res = vars(parser.parse_args())
    # This code doesn't work!
    # if parse_res['all']:
    #     if (parse_res['input'] is not None):
    #         print(parser.print_help())
    #         print("Please specify either input object name or all inputs")
    #         exit(1)
    #     for subfolder in dir(Inputs):
    #         for inp in dir(getattr(Inputs, subfolder)):
    #             main("a.b", input_to_repr(getattr(getattr(Inputs, subfolder), inp), parse_res['c'], parse_res['d']))
    # else:
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
                        main(first_name + "." + last_name, input_to_repr(getattr(getattr(Inputs, subfolder), inp), parse_res['c'], parse_res['d'], parse_res['clist']))
