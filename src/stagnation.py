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
        J = I #deepcopy_DNF(I)
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


def checkAreaAroundStuck (I, repr, samplepoints):
    ns = getNeighbors (repr, I)
    # print(ns) #Debug 
    LII = dnfconjunction( list3D_to_listof2Darrays(I), repr.get_affineSubspace(), 0)
    costI = (cost(LII, samplepoints))[0]
    costLists = [ [costI] ]
    
    for _ in range(1, 2):
        nextneighbors = []
        neighborcostList = []
        for N in ns:
            # print(N) 
            S = getNeighbors(repr, N)
            nextneighbors = nextneighbors + S
            LII = dnfconjunction( list3D_to_listof2Darrays(N), repr.get_affineSubspace(), 0)
            neighborcostList = neighborcostList + [(cost(LII, samplepoints))[0]] 
        ns = nextneighbors.copy
        costLists.append(neighborcostList)
    
    return costLists   