import random
from math import log, e
import copy

from repr import Repr
from cost_funcs import cost
from guess import SearchSpaceNeighbors
from dnfs_and_transitions import list3D_to_listof2Darrays, dnfconjunction, dnfnegation
from print import initialized, statistics, z3statistics, invariantfound, timestatistics, prettyprint_samplepoints, noInvariantFound
from print import SAexit, SAsuccess, n2plotter, SAfail, print_with_mode, list_to_string, printTemperaturePrompt

from configure import Configure as conf  

# Global variable for the number of candidates
C = 10

def geneticProgramming(inputname, repr: Repr, I_list, samplepoints, process_id, return_value, GA_Gamma, z3_callcount, costTimeLists, output):
    random.seed()
    #IMPORTANT: Make sure only 1 thread; as there is no parallelism.


    n = repr.get_n()
    tmax = repr.get_tmax()

    # Initialize population
    population = []
    for _ in range(C):
        I = random.choice(I_list)
        LII = dnfconjunction(list3D_to_listof2Darrays(I), repr.get_affineSubspace(), 0)
        (costI, costlist) = cost(LII, samplepoints)
        population.append((I, costI, costlist))

    for t in range(1, tmax + 1):
        if (t % conf.NUM_ROUND_CHECK_EARLY_EXIT == 0):
            for i in range(conf.num_processes):
                if return_value[i] is not None and return_value[i][0] is not None:
                    return_value[process_id] = (None, t)
                    SAexit(process_id, repr.get_colorslist(), output)
                    I_list[process_id] = population[0][0]
                    return

        # Sort population by cost
        population.sort(key=lambda x: x[1])
        
        if population[0][1] == 0:
            return_value[process_id] = (population[0][0], t)
            SAsuccess(process_id, repr.get_colorslist(), output)
            I_list[process_id] = population[0][0]
            return

        new_population = population[:C//2]  # Keep the best half

        while len(new_population) < C:
            # Select two parents
            parents = random.sample(population[:C//2], 2)
            parent1, parent2 = parents[0][0], parents[1][0]

            # Mutation: Select any neighbor of the parents
            child1 = mutate(parent1, repr, n)
            child2 = mutate(parent2, repr, n)

            # Add mutated children to new population
            for child in [child1, child2]:
                LII = dnfconjunction(list3D_to_listof2Darrays(child), repr.get_affineSubspace(), 0)
                (costI, costlist) = cost(LII, samplepoints)
                new_population.append((child, costI, costlist))

        population = new_population

        for candidate in population:
            I, costI, costlist = candidate
            descent = 1 if costI < population[0][1] else 0
            reject = 0 if descent == 1 else 1
            statistics(process_id, t, I, costI, descent, reject, costlist, -1, repr.get_Var(), repr.get_colorslist(), output)

    # Process 'process_id' Failed!
    SAfail(process_id, repr.get_colorslist(), output)
    return_value[process_id] = (None, t)
    I_list[process_id] = population[0][0]
    return

def mutate(parent, repr, n):
    neighbors = SearchSpaceNeighbors(parent, repr, repr.get_d(), repr.get_cList(), repr.get_k1(), n)
    if neighbors:
        P = random.choice(neighbors)
        child = copy.deepcopy(parent)
        child[P[0]][P[1]] = P[2]
        return child
    else:
        return parent
