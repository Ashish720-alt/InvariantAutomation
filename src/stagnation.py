from configure import Configure as conf
from dnfs_and_transitions import deepcopy_DNF, dnfconjunction, list3D_to_listof2Darrays
from cost_funcs import cost
from repr import Repr
from guess import getNewRotConstant, getNewTranslationConstant
import numpy as np

def listof2Darrays_to_list3D (I):
    A = [cc.tolist() for cc in I ]
    return A


def getNeighbors (repr: Repr, I):
    n = repr.get_n()
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
                constlist = getNewRotConstant(oldcoeff, oldconst, r, repr.get_k1())
                for c in constlist:
                    neighbors.append( ( i, j, r + [-1,c]) )
            transconslist = getNewTranslationConstant(oldcoeff, oldconst, repr.get_k1())
            for c in transconslist:
                neighbors.append( ( i, j, oldcoeff + [-1,c]) )
    
    neighbor_true = []
    for (i,j, p) in neighbors:
        J = listof2Darrays_to_list3D (deepcopy_DNF(I))
        J[i][j] = p
        neighbor_true.append(J)
    
    return neighbor_true


def isStagnant( costlistTime):
    if (len(costlistTime) < conf.STAGNANT_TIME_WINDOW):
        return False
    M = max(costlistTime)
    m = min(costlistTime)
    return (M - m <= conf.STAGNANT_COST_RANGE)

def checkLocalMinima (I, repr, samplepoints):
    neighbors = getNeighbors (repr, I)
    neighborcostList = []
    LII = dnfconjunction( list3D_to_listof2Darrays(I), repr.get_affineSubspace(), 0)
    costI = (cost(LII, samplepoints))[0]
    for N in neighbors:
        LII = dnfconjunction( list3D_to_listof2Darrays(N), repr.get_affineSubspace(), 0)
        neighborcostList = neighborcostList + [(cost(LII, samplepoints))[0]]    
    return (costI <= min(neighborcostList))    

#Ineffective function, 1 neighbors are around 200-300 and 2nd neighbors are around 17,000
def checkAreaAroundStuck (I, repr, samplepoints):
    ns = getNeighbors (repr, I)
    LII = dnfconjunction( list3D_to_listof2Darrays(I), repr.get_affineSubspace(), 0)
    costI = (cost(LII, samplepoints))[0]
    costLists = [ [costI] ]
    
    for _ in range(conf.STAGNATION_AREA_CHECK):
        nextneighbors = []
        neighborcostList = []
        for N in ns:
            S = getNeighbors(repr, N)
            nextneighbors = nextneighbors + S
            LIN = dnfconjunction( list3D_to_listof2Darrays(N), repr.get_affineSubspace(), 0)
            neighborcostList.append(cost(LIN, samplepoints)[0]) 
        ns = nextneighbors.copy()
        costLists.append(neighborcostList)
    
    return costLists   

def gradientdescent(I, repr, samplepoints, color):    
    def get_bestNeighbor(repr, I, samplepoints):
        def getcostI(I, repr, samplepoints):
            return (cost(dnfconjunction( list3D_to_listof2Darrays(I), repr.get_affineSubspace(), 0), samplepoints))[0] 

        neighbors = getNeighbors(repr, I)

        if not neighbors:
            return None 
        costs = [getcostI(neighbor, repr, samplepoints) for neighbor in neighbors]
        min_cost_index = costs.index(min(costs))
        return (neighbors[min_cost_index], costs[min_cost_index])    
    
    c = (cost(dnfconjunction( list3D_to_listof2Darrays(I), repr.get_affineSubspace(), 0), samplepoints))[0] 
    
    #Don't want to do this for rare types of stagnation.
    if (c > 1000):
        return
    
    print(color + "Gradient Descent starts ...")
    
    
    t = 0
    while (t <= 1000):
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



    print(color + "Gradient Descent ends.")
    return            
    
    