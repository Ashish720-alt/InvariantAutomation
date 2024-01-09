from configure import Configure as conf
from dnfs_and_transitions import deepcopy_DNF
from cost_funcs import cost


def isStagnant( costlistTime):
    if (len(costlistTime) < conf.STAGNANT_TIME_WINDOW):
        return False
    M = max(costlistTime)
    m = min(costlistTime)
    return (M - m <= conf.STAGNANT_COST_RANGE)

def checkLocalMinima (I, neighbors, samplepoints):
    neighborcostList = []
    J = deepcopy_DNF(I)
    costI = (cost(J, samplepoints))[0]
    for (i,j, p) in neighbors:
        J[i][j] = p
        neighborcostList = neighborcostList + [(cost(J, samplepoints))[0]]
        J[i][j]  = I[i][j]
    
    return (costI <= min(neighborcostList))    