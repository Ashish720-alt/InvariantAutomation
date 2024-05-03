
import numpy as np
from dnfs_and_transitions import dnfTrue, dnfconjunction
from configure import Configure as conf

def list_union(X, Y):
    return list(set(X + Y))

def extract_constants(P,B,T,Q):
    n = len(P[0][0]) - 2
    def extractdnfconsts(dnf):
        (const, coeff) = (set([0,1]) , set([0,1]))
        for cc in dnf:
            for p in cc:
                const |= set([p[n+1]])
                coeff |= set(p[:-2].flatten())
        return (coeff, const)
        
    def extractBtrconstants(Btr):
        def extractptfconsts(ptf):
            (const, coeff) = (set() , set())
            for row in ptf:
                    const |= set([row[n]])
                    coeff |= set(row[:-1].flatten())  
            return (coeff, const)
        (coeff, const) = extractdnfconsts(Btr.b)
        for ptf in Btr.tlist:
            (coeff_ptf, const_ptf) = extractptfconsts(ptf)
            (coeff, const) = (coeff | coeff_ptf, const | const_ptf)
        return (coeff, const)         

    coeff = extractdnfconsts(P)[0] | extractdnfconsts(B)[0] | extractdnfconsts(Q)[0]
    const = extractdnfconsts(P)[1] | extractdnfconsts(B)[1] | extractdnfconsts(Q)[1]
    for Btr in T:
        (coeff_btr, const_btr) = extractBtrconstants(Btr)
        (coeff, const) = (coeff | coeff_btr, const | const_btr)
    return (list(coeff), list(const))


def ccl(X):
    return list(range(min(X), max(X)+1, 1))

def gd_coeff(X):
    return list_union(X, [(-1)*x for x in X])   

def gd_const(X):
    temp = list_union(X, [(-1)*x for x in X])
    return list_union(temp, [t-1 for t in temp])

def scd(k):
    return list(range(-k, k+1, 1))

def npcd_const(X, r):
    ret = X
    for i in range(1, r+1):
        ret = ret + [x+i for x in X] +  [x-i for x in X]
    return list(set(ret))


def D_singlecoeff(coeff):
    return ccl(gd_coeff(list_union(scd(conf.coeff_k), coeff)))

def D_const(pc):
    return ccl(gd_const(list_union(scd(conf.coeff_k), npcd_const(pc, conf.coeff_r) )))

def D_p(P, B, T , Q):
    (coeff, const) = extract_constants(P,B,T,Q)
    
    
    pc = list_union(coeff, const)
    return pc
    
    # return (D_singlecoeff(coeff) , D_const(pc) )

# Testing: 
# import repr
# P = [np.array([[1, 0, 3]])]
# B = [np.array([[1, -1, 5]])]
# Q = [np.array([[1, 0, 6]])]

# class B_LItransitionrel:
#     def __init__(self, transition_matrix_list, DNF, B):
#         self.tlist = transition_matrix_list
#         self.b = dnfconjunction(DNF, B, gLII = 1)

# def genLItransitionrel(B, *args):
#     return [B_LItransitionrel(x[0], x[1], B) for x in args ]

# T = genLItransitionrel(B, ( [np.array([[1, 1], [0, 1]])] , dnfTrue(1) ) ) 

# S = extract_constants(P, B, T, Q)
# print(gd_coeff(S[0]), gd_const(S[1]))
# print(scd(5), npcd_const([1,2,3,1000], 2))
# print(D_singlecoeff(S[0]), D_const(list_union(S[0], S[1])))
# print(D_p(P, B, T, Q))