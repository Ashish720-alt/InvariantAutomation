import random
from math import log, e

from repr import Repr
from cost_funcs import cost
from guess import SearchSpaceNeighbors  
from dnfs_and_transitions import  list3D_to_listof2Darrays, dnfconjunction, dnfnegation
from print import initialized, statistics, z3statistics, invariantfound, timestatistics, prettyprint_samplepoints, noInvariantFound
from print import SAexit, SAsuccess, n2plotter, SAfail, print_with_mode, list_to_string, printTemperaturePrompt

from configure import Configure as conf  # Import configuration settings

def gradientDescent(inputname, repr: Repr, I_list, samplepoints, process_id, return_value, GD_Gamma, z3_callcount, costTimeLists, output):

    #IMPORTANT: Make sure only 1 thread; as there is no randomization.

    I = I_list[process_id]
    n = repr.get_n()
    tmax = repr.get_tmax()
    LII = dnfconjunction(list3D_to_listof2Darrays(I_list[process_id]), repr.get_affineSubspace(), 0)
    (costI, costlist) = cost(LII, samplepoints)


    for t in range(1, tmax + 1):
        if (t % conf.NUM_ROUND_CHECK_EARLY_EXIT == 0):
            for i in range(conf.num_processes):
                if return_value[i] is not None and return_value[i][0] is not None:
                    return_value[process_id] = (None, t)
                    GDexit(process_id, repr.get_colorslist(), output)
                    I_list[process_id] = I
                    return

        if costI == 0:
            return_value[process_id] = (I, t)
            SAsuccess(process_id, repr.get_colorslist(), output)
            I_list[process_id] = I
            return

        neighbors = SearchSpaceNeighbors(I, repr, repr.get_d(), repr.get_cList(), repr.get_k1(), n)
        best_neighbor = None
        best_cost = float('inf')
        for P in neighbors:
            oldpredicate = I[P[0]][P[1]]
            I[P[0]][P[1]] = P[2]
            LII = dnfconjunction(list3D_to_listof2Darrays(I), repr.get_affineSubspace(), 0)
            (costInew, costlist) = cost(LII, samplepoints)
            if costInew < best_cost:
                best_cost = costInew
                best_neighbor = P
            I[P[0]][P[1]] = oldpredicate

        if best_neighbor is not None and best_cost < costI:
            P = best_neighbor
            I[P[0]][P[1]] = P[2]
            costI = best_cost
            descent = 1
            reject = True
        else:
            descent = 0
            reject = True

        statistics(process_id, t, I, costI, descent, reject, costlist, -1, repr.get_Var(), repr.get_colorslist(), output)

    # Process 'process_id' Failed!
    SAfail(process_id, repr.get_colorslist(), output)
    return_value[process_id] = (None, t)
    I_list[process_id] = I
    return
