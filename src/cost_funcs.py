
""" Cost Functions.
"""

import numpy as np
from dnfs_and_transitions import dnfnegation
from configure import Configure as conf
from copy import deepcopy

class setType:
    plus = "plus"
    minus = "minus"
    ICE = "ICE"




def LIPptdistance(p, pt):
    return max( sum(p[:-2]* pt) - p[-1] , 0)

def negationLIpredicate(p):
    return (dnfnegation( [np.array( [p], ndmin = 2 )] ))[0][0]

def d(p, pt, pt_type):
    if (pt_type == setType.plus):
        return LIPptdistance(p, pt)
    elif (pt_type == setType.minus):
        return LIPptdistance(negationLIpredicate(p), pt)
    else:
        return min( LIPptdistance(negationLIpredicate(p), pt[0]) , LIPptdistance(p, pt[1])  )

def U(r, U_type):
    if (U_type == setType.plus):
        return 1.0
    elif (U_type == setType.minus):
        return 1.0
    else:
        return 1.0 

def cost(costlist):
    return sum ([ min([ sum([p for p in cc]) for cc in pt_I]) for pt_I in costlist ])

def costtuple(I, S, set_type ):
    if (set_type == setType.plus):
        costlist = [ [ [LIPptdistance(p, pt) for p in cc  ] for cc in I ]  for pt in S]
        return (cost(costlist), costlist)
    elif (set_type == setType.minus):
        costlist = [ [ [LIPptdistance(negationLIpredicate(p), pt) for p in cc  ] for cc in I ]  for pt in S] 
        return (cost(costlist), costlist)
    else:
        costlist = ([ [ [LIPptdistance(negationLIpredicate(p), pt[0]) for p in cc  ] for cc in I ]  for pt in S], 
                            [ [ [LIPptdistance(p, pt[1]) for p in cc  ] for cc in I ]  for pt in S] )
        return ( min( cost(costlist[0]), cost(costlist[1]) ), costlist)    


def optimized_costtuple(I, S, set_type, prev_costlist, inv_i):
    costlist = deepcopy(prev_costlist)
    if (set_type == setType.plus or set_type == setType.minus):
        for j,pt in enumerate(S):
            costlist[j][inv_i[0]][inv_i[1]] = d(I[inv_i[0]][inv_i[1]], pt, set_type)
        return (cost(costlist), costlist)
    else:
        for j,pt in enumerate(S):
            costlist[0][j][inv_i[0]][inv_i[1]] = d(I[inv_i[0]][inv_i[1]], pt, set_type)        
            costlist[1][j][inv_i[0]][inv_i[1]] = d(I[inv_i[0]][inv_i[1]], pt, set_type) 
        return ( min( cost(costlist[0]), cost(costlist[1]) ), costlist) 

def f1(costplus, costminus, costICE):
    K = conf.alpha/3.0
    gamma = conf.gamma
    Uplus = U(costplus, setType.plus)
    Uminus = U(costminus, setType.minus)
    UICE = U(costICE, setType.ICE)
    return   ((K *  gamma**(-costplus) )/ Uplus) + ((K *  gamma**(-costminus) )/ Uminus) + ((K *  gamma**(-costICE) )/ UICE)

def f2(costplus, costminus, costICE, beta  ):
    K = conf.alpha
    gamma = conf.gamma
    exp = -beta * (costplus + costminus + costICE)
    
    return K * (gamma**exp / 1.0)    

def cost_to_f(costplus, costminus, costICE, beta):
    return f2(costplus, costminus, costICE, beta)

# i is the invariant index to change as a tuple: (cc_index, pred_index) , where cc_index and pred_index start from 0
def f(I, tupleofpoints, beta , prev_costtuple = ([], [], []), i = () ):
    if (i == () ):
        (costplus, costplus_list) = costtuple(I, tupleofpoints[0], setType.plus)
        (costminus, costminus_list) = costtuple(I, tupleofpoints[1], setType.minus)
        (costICE, costICE_list) = costtuple(I, tupleofpoints[2], setType.ICE)
    else:
        (costplus, costplus_list) = optimized_costtuple(I, tupleofpoints[0], setType.plus, prev_costtuple[0], i)
        (costminus, costminus_list) = optimized_costtuple(I, tupleofpoints[1], setType.minus, prev_costtuple[1], i)
        (costICE, costICE_list) = optimized_costtuple(I, tupleofpoints[2], setType.ICE, prev_costtuple[2], i)        
    
    return (cost_to_f(costplus, costminus, costICE, beta), costplus + costminus + costICE, (costplus_list , costminus_list, costICE_list))




# # Testing
# p = np.array([1, 2, 3, 4, -1, 3])
# print(negationLIpredicate(p))
# pt = [1, 1, 1, 1]
# print(LIPptdistance(p, pt))

# I = [ np.array([[3, -1, 6] ])  ]
# plus = [ [0] ]
# minus = [ [7], [10000] ]
# ICE = [ ( [5] , [6]  )  ]
# (prev_f, prev_cost, prev_costtuple) = f(I, (plus, minus, ICE))
# print (prev_f, prev_cost, prev_costtuple)
# index = [0, 0]
# Inew = I
# Inew[0][0][2] = 6
# print( f(Inew, (plus, minus, ICE) , prev_costtuple, index) )


