""" Functions that convert formulas in matrix representation to z3 expression.
DNF is represented as matrices.
"""

import numpy as np
from configure import Configure as conf
from z3 import *
from itertools import product

#IMP: Note that there is a function in cdd python library to remove rendundancies within a cc.

# dnf's are lists of 2D numpy arrays, rather than 3D numpy arrays.

def dnfTrue (n):
    P = np.zeros(shape = (n+2))
    P[n+1] = -1
    return [ np.array([P]) ] 

def dnfFalse (n):
    P = np.zeros(shape = (n+2))
    P[n+1] = -1
    P[n+2] = -1
    return [ np.array([P]) ] 


# It always returns LII
def dnfnegation (dnf):
    dnf_LII = genLII_to_LII(dnf) #We have to do this here, as negation of an equality LI predicate is a disjunction of two LI predicates.
    n = len(dnf_LII[0][0]) - 2

    def dnfnegation_LIp_to_LIp (p, n):
        p_neg = p
        p_neg[n] = 2
        p_neg = p_neg * -1
        p_neg[n+1] = p_neg[n+1] - 1
        return p_neg
    
    d = len(dnf_LII)
    dnf_c_list = []
    negdnf_dspace = []
    for cc in dnf_LII:
        cc_length = len(cc)
        dnf_c_list.append(cc_length)
        negdnf_dspace = list(product(negdnf_dspace, range(cc_length) ))
    
    negdnf = []
    for it in negdnf_dspace:
        cc = np.empty( shape=(0, n + 2), dtype = int )
        for j in range(d):
            p = dnfnegation_LIp_to_LIp(dnf_LII[j][it[j]], n)
            cc = np.concatenate((cc, np.array([p], ndmin=2)))
        negdnf.append(cc)

    return negdnf


def dnfconjunction (dnf1, dnf2, gLII):    
    ret = []
    for cc1 in dnf1:
        for cc2 in dnf2:
            cc = np.append(cc1, cc2, axis = 0)
            ret.append(cc)   
    if (gLII == 0):
        ret = genLII_to_LII(ret)
    return ret

def dnfdisjunction (dnf1, dnf2, gLII):
    ret = dnf1 + dnf2
    if (gLII == 0):
        ret = genLII_to_LII(ret)
    return ret

#Converts generalized LI invariant to LI invariant
def genLII_to_LII (genLII):
    n = len(genLII[0][0]) - 2 
    LII = []
    for gencc in genLII:
        cc = np.empty(shape=(0, n + 2 ), , dtype = int)
        for genp in gencc:
            p = genp
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
    return ret

# x is a python list, ptf is a np array of dimension 2, and return type is python list
def transition ( x , ptf):
    xmatrix = np.concatenate((np.array(x), np.array([1])))
    ymatrix_nparray = np.dot(pt_matrix, np.transpose(ptf))
    ymatrix_list = ymatrix_nparray.tolist()
    y = ymatrix_list[:-1]
    return y


#Testing
#print( DNFconjunction( [np.array([[1,2,3,-1,1], [1,2,3,-1,2]]) , np.array([[1,1,1,2,1], [1,2,2,2,2]]) ], [np.array([[1,2,3,-1,1], [1,2,3,-1,2]])] )  )


def DNF_to_z3expr(m, p=''):
    if np.size(m) == 0:
        return True

    d0 = len(m)
    d1 = len(m[0])
    d2 = len(m[0][0])
    return simplify(  Or([
        And([
            conf.OP[int(m[i][j][-2])](
                Sum([
                    m[i][j][k] * Int(('x%s'+p) % k)
                    for k in range(d2-2)
                ]),
                int(m[i][j][-1])
            )
            for j in range(d1)
        ])
        for i in range(d0)
    ]))


def DNF_to_z3expr_p(m):
    """ Get a prime version of the DNF.
    """
    return DNF_to_z3expr(m, 'p')


def trans_matrix_to_z3expr(A):
    """ 
    A is a (n+1) * (n+1) matrix.
    """
    d = len(A)
    return simplify(And([
        Int("x%sp" % i) ==
        Sum([
            int(A[i][j]) * Int("x%s" % j)
            for j in range(d-1)
        ]) +
        int(A[i][d-1])
        for i in range(d-1)
    ]))


def trans_func_to_z3expr(f):
    ret = True
    for i in range(len(f)-1, -1, -1):
        ret = If(DNF_to_z3expr(f[i].b),
                 trans_matrix_to_z3expr(f[i].t),
                 ret)
    return ret

# Testing the functions.
# A = np.array([[1, 2, 3, 1], [2, 3, 1, 4], [1, 3, 1, 4], [0, 0, 0, 1]], ndmin=2)
# X = trans_matrix_to_z3expr(A)
# print(X, X)
# S = total_transition_function(A)
# print(S[0].b, S[0].t)


def op_norm_pred(P):
    n = len(P) - 2
    if (P[n] > 0):
        return np.array(np.multiply(P, -1), ndmin=2)
    elif (P[n] in [0, 10, -10] ):
        temp1, temp2 = np.multiply(P, -1), P
        if (P[n] == 0):
            temp1[n] = -1
            temp2[n] = -1
        else:
            temp1[n] = -2
            temp2[n] = -2            
        return np.array([temp1, temp2], ndmin=2)
    return np.array(P, ndmin=2)


def op_norm_conj(C):
    assert(len(C) > 0)  # assuming C not empty
    return np.concatenate([op_norm_pred(C[i]) for i in range(len(C))])


def norm_disj(D, conjunct_size):
    n = len(D) - 2
    if (n == 0):
        return np.empty(shape=(0, conjunct_size, n+2), dtype=int)
    C = op_norm_conj(D[0])
    
    padding_pred = np.zeros(n + 2)
    padding_pred[n] = -1
    while (len(C) <= conjunct_size):
        C = np.concatenate((C, np.array(padding_pred, ndmin=2)))
    return np.concatenate((np.array(C, ndmin=3), norm_disj(D[1:], conjunct_size)))


def norm_DNF(D, n):
    """
    :n: The number of variables.
    """
    # Let operator_normalized versions only have <= or < operators.

    max_size = 0
    for C in D:
        curr_size = 0
        for P in C:
            curr_size = curr_size + (2 if (P[n] == 0) else 1)
        max_size = max(max_size, curr_size)
    return norm_disj(D, max_size)
