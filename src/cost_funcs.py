
""" Cost Functions.
This module includes cost functions.
"""

import numpy as np
from dnfs_and_transitions import dnfnegation
from configure import Configure as conf

class setType:
    plus = "plus"
    minus = "minus"
    ICE = "ICE"

def LIPptdistance(p, pt):
    return max(sum(p[:-2]* pt) - p[-1] , 0)

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
        return 1
    elif (U_type == setType.minus):
        return 1
    else:
        return 1   

def mincost(mincostlist):
    return sum ([ min([ sum([p for p in cc]) for cc in pt_I]) for pt_I in mincostlist ])

def mincosttuple(I, S, set_type ):
    mincostlist = [ [ [d(p, pt, set_type) for p in cc  ] for cc in I ]  for pt in S]
    r = mincost(mincostlist)
    return (r, U(r, set_type), mincostlist)


def optimized_mincosttuple(I, S, set_type, prev_mincostlist, inv_i):
    mincostlist = prev_mincostlist
    
    for j,pt in enumerate(S):
        mincostlist[j][inv_i[0]][inv_i[1]] = d(I[inv_i[0]][inv_i[1]], pt, set_type)
    r =  mincost(mincostlist)
    return (r, U(r, set_type), mincostlist)   

# i is the invariant index to change as a tuple: (cc_index, pred_index) , where cc_index and pred_index start from 0
def cost(I, tupleofpoints, prev_mincosttuple = ([], [], []), i = () ):
    K = conf.alpha/3.0
    gamma = conf.gamma
    
    if (i == () ):
        (mincostplus, Uplus, mincostplus_list) = mincosttuple(I, tupleofpoints[0], setType.plus)
        (mincostminus, Uminus, mincostminus_list) = mincosttuple(I, tupleofpoints[1], setType.minus)
        (mincostICE, UICE, mincostICE_list) = mincosttuple(I, tupleofpoints[2], setType.ICE)
    else:
        (mincostplus, Uplus, mincostplus_list) = optimized_mincosttuple(I, tupleofpoints[0], setType.plus, prev_mincosttuple[0], i)
        (mincostminus, Uminus, mincostminus_list) = optimized_mincosttuple(I, tupleofpoints[1], setType.minus, prev_mincosttuple[1], i)
        (mincostICE, UICE, mincostICE_list) = optimized_mincosttuple(I, tupleofpoints[2], setType.ICE, prev_mincosttuple[2], i)        
    
    A = ((K *  gamma**(-mincostplus) )/ Uplus)
    B = ((K *  gamma**(-mincostminus) )/ Uminus)
    C = ((K *  gamma**(-mincostICE) )/ UICE)
    cost = A + B + C # HERE DeBugging
    # print('\t', (mincostplus, mincostminus, mincostICE), (A, B, C) )
    return (cost, mincostplus + mincostminus + mincostICE, (mincostplus_list , mincostminus_list, mincostICE_list))




# # Testing
# p = np.array([1, 2, 3, 4, -1, 3])
# print(negationLIpredicate(p))
# pt = [1, 1, 1, 1]
# print(LIPptdistance(p, pt))

# I = [ np.array([[3, -1, 6] ])  ]
# plus = [ [0] ]
# minus = [ [7], [10000] ]
# ICE = [ ( [5] , [6]  )  ]
# (prev_cost, prev_mincost, prev_mincosttuple) = cost(I, (plus, minus, ICE))
# print (prev_cost, prev_mincost, prev_mincosttuple)
# index = [0, 0]
# Inew = I
# Inew[0][0][2] = 6
# print( cost(Inew, (plus, minus, ICE) , prev_mincosttuple, index) )


