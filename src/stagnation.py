from configure import Configure as conf
from dnfs_and_transitions import  dnfconjunction, list3D_to_listof2Darrays
from cost_funcs import cost
from repr import Repr
from guess import SearchSpaceNeighbors

#Checks if cost changes for some time are within some window, if so calls it as stagnation!
def isStagnant( costlistTime):
    if (len(costlistTime) < conf.STAGNANT_TIME_WINDOW):
        return False
    M = max(costlistTime)
    m = min(costlistTime)
    return (M - m <= conf.STAGNANT_COST_RANGE)

# Simply uses gradient descent from current invariant till it hits a local minima 
def checkLocalMinima(I, repr, samplepoints, color):    
    def get_bestNeighbor(repr, I, samplepoints):
        def getcostI(I, repr, samplepoints):
            return (cost(dnfconjunction( list3D_to_listof2Darrays(I), repr.get_affineSubspace(), 0), samplepoints))[0] 
        neighbors = SearchSpaceNeighbors(I, repr, repr.get_d() , repr.get_c() , repr.get_k1() , repr.get_n())
        if not neighbors:
            return None 
        costs = [getcostI(neighbor, repr, samplepoints) for neighbor in neighbors]
        min_cost_index = costs.index(min(costs))
        return (neighbors[min_cost_index], costs[min_cost_index])    
    c = (cost(dnfconjunction( list3D_to_listof2Darrays(I), repr.get_affineSubspace(), 0), samplepoints))[0] 
    
    # if (c > 1000): #Don't want to do this for rare types of stagnation.
    #     return
    
    
    print(color + "Gradient Descent starts ...")
    t = 0
    while (t <= conf.LOCAL_MINIMA_DEPTH_CHECKER):
        (Inew, cnew) = get_bestNeighbor(repr, I, samplepoints)
        if (cnew >= c):
            print(color + str(t), "Hit a local minima at previous invariant!")
            print(color + "Gradient Descent ends.")
            return
        else:
            c = cnew
            I = Inew
            print(color + str(t), I, c)
        t = t + 1
    print(color + "No local minima found for" + str(conf.LOCAL_MINIMA_DEPTH_CHECKER) + " steps. Gradient Descent ends.")
    return            

