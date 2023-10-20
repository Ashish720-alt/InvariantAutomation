""" Functions that convert formulas in matrix representation to z3 expression.
DNF is represented as matrices.
"""

import numpy as np
import copy
from z3 import *
from itertools import product

#IMP: Note that there is a function in cdd python library to remove rendundancies within a cc.

# dnf's are lists of 2D numpy arrays, rather than 3D numpy arrays.

def dnfTrue (n):
    p = np.zeros(shape = (n+2), dtype = int)
    p[n] = -1
    return [ np.array([p], ndmin = 2) ] 


def dnfFalse (n):
    p = np.zeros(shape = (n+2), dtype = int)
    p[n] = -1
    p[n+1] = -1
    return [ np.array([p]) ] 

#Converts generalized LI invariant to LI invariant
def genLII_to_LII (genLII):
    n = len(genLII[0][0]) - 2 
    LII = []
    for gencc in genLII:
        cc = np.empty(shape=(0, n + 2 ), dtype = int)
        for genp in gencc:
            p = copy.deepcopy(genp)
            if (p[n] == -2):
                p[n] = -1
                p[n+1] = p[n+1] - 1
                cc = np.concatenate((cc, np.array([p], ndmin=2))) 
            elif (p[n] == -1):
                cc = np.concatenate((cc, np.array([p], ndmin=2))) 
            elif (p[n] == 1):
                p = p * -1
                cc = np.concatenate((cc, np.array([p], ndmin=2))) 
            elif (p[n] == 2):
                p = p * -1
                p[n+1] = p[n+1] - 1
                cc = np.concatenate((cc, np.array([p], ndmin=2))) 
            elif (p[n] == 0):
                p2 = p * -1
                p[n] = -1
                p2[n] = -1
                cc = np.concatenate((cc, np.array([p, p2], ndmin=2)))
        LII.append(cc)               
    return deepcopy_DNF(LII)


# It always returns LII
def dnfnegation (dnf):
    dnf_LII = genLII_to_LII(dnf) #We have to do this here, as negation of an equality LI predicate is a disjunction of two LI predicates.
    n = len(dnf_LII[0][0]) - 2

    def dnfnegation_LIp_to_LIp (p, n):
        p_neg = p
        p_neg[n] = 2
        p_neg = p_neg * -1
        p_neg[n+1] = p_neg[n+1] - 1
        p_neg[n] = -1
        return p_neg
    
    d = len(dnf_LII)
    dnf_c_list_of_lists = [] 
    for cc in dnf_LII:
        dnf_c_list_of_lists.append(range(len(cc)))

    negdnf = []
    for it in product(*dnf_c_list_of_lists):
        cc = np.empty( shape=(0, n + 2), dtype = int )
        for j in range(d):
            p = dnfnegation_LIp_to_LIp(dnf_LII[j][it[j]], n)
            cc = np.concatenate((cc, np.array([p], ndmin=2)))
        negdnf.append(cc)
    return deepcopy_DNF(negdnf)

def dnfconjunction (dnf1, dnf2, gLII):    
    ret = []
    if (len(dnf1) == 0):
        ret = dnf2
    elif (len(dnf2) == 0):
        ret = dnf1
    else:
        for cc1 in dnf1:
            for cc2 in dnf2:
                cc = np.append(cc1, cc2, axis = 0)
                ret.append(cc)   
    if (gLII == 0):
        ret = genLII_to_LII(ret)
    return deepcopy_DNF(ret)

def dnfdisjunction (dnf1, dnf2, gLII):
    ret = dnf1 + dnf2
    if (gLII == 0):
        ret = genLII_to_LII(ret)
    return deepcopy_DNF(ret)


    
    

# x is a python list, ptf is a np array of dimension 2, and return type is a python list
def transition ( x , ptf):
    xmatrix = np.concatenate((np.array(x), np.array([1])))
    ymatrix_nparray = np.dot(xmatrix, np.transpose(ptf))
    ymatrix_list = ymatrix_nparray.tolist()
    y = ymatrix_list[:-1]
    return y

def DNF_aslist(I):
    I_list = []
    for cc in I:
        I_list.append(cc.tolist())
    return I_list

def deepcopy_DNF(I):
    n = len(I[0][0]) - 2    
    I_new = []
    for cc in I:
        cc_new = np.empty( shape=(0, n + 2), dtype = int )
        for p in cc:
            cc_new = np.concatenate((cc_new, np.array([copy.deepcopy(p)], ndmin=2)))
        I_new.append(cc_new)
    return I_new


# rtpred is ( n-rotationlist, n-translationlist ), rtcc is list of rtpreds, rtinv is list of rtcc's
def RTI_to_LII(rtinv):
    def rtcc_to_LIcc(rtcc, n):
        def rtpred_to_LIpred(rtpred):
            coeff = rtpred[0]
            const =  np.dot(np.asarray(rtpred[0]), np.asarray(rtpred[1])) 
            return np.array(coeff + [-1, const])
        LIcc = np.empty(shape=(0, n + 2 ), dtype = int)
        for rtpred in rtcc: 
            LIcc = np.concatenate((LIcc, np.array( rtpred_to_LIpred(rtpred) , ndmin=2) )) 
        return LIcc
    n = len(rtinv[0][0][0])
    return [rtcc_to_LIcc(rtcc, n) for rtcc in rtinv]

def list3D_to_listof2Darrays (I):
    def cclist_to_ccarray (cc_I):
        return np.array([np.array(p) for p in cc_I])
    A = [cclist_to_ccarray(cc) for cc in I ]
    return A

# def removestateduplicates (l):
#     temp = list({tuple(x) for x in l})
#     return [list(x) for x in temp]
    
# def removeICEpairduplicates (l):
#     if (len(l) == 0):
#         return []
#     n = len(l[0][0])
#     temp = list({tuple(x[0] + x[1]) for x in l})
#     temp2 = [list(x) for x in temp]
#     return [ (x[0:n] , x[n:]  ) for x in temp2  ]
    

# print( removestateduplicates( [ [1,2] , [3,4] , [5,6] , [7,8] , [8,9] , [1,2]  ]) )
# print( removeICEpairduplicates( [ ([1,2] , [3,4]) , ([5,6] , [7,8]) , ([8,9] , [1,2]) , ([8,9] , [1,2])  ]) )

# # Testing
# print(dnfTrue(2))
# print(dnfFalse(2))
# dnfTest = [np.array([[1,2,1,0,3], [1,1,3,1,2]], ndmin = 2), np.array([[1,2,1,0,3], [1,1,3,0,2]], ndmin = 2)]
# print(genLII_to_LII( dnfTest ))
# dnfTest1 = [np.array([[1,2,1,-1,2] , [3, 1, 2, 1,1]], ndmin = 2) , np.array([[1,2,1,2,2] , [3, 1, 2, 1,1]], ndmin = 2) ]
# print(dnfnegation(dnfTest1))
# print( dnfconjunction( [np.array([[1,2,3,-1,1], [1,2,3,0,2]]) , np.array([[1,1,1,2,1], [1,2,2,2,2]]) ], [np.array([[1,2,3,-1,1], [1,2,3,-1,2]])] , 0 )  )
# print( dnfdisjunction( [np.array([[1,2,3,-1,1], [1,2,3,0,2]]) , np.array([[1,1,1,2,1], [1,2,2,2,2]]) ], [np.array([[1,2,3,-1,1], [1,2,3,-1,2]])] , 1 )  )
# print(transition([1,5,3] , np.array([[1, 0, 0, 1], [0, 2, 0, 0], [1, 1, 0, 0] , [0, 0, 0, 1]  ]) ))
# print( rtinv_to_LIinv( [[ ([1,3],[6,2])  ]]  ))






