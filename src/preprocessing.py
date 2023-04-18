import numpy as np
from z3 import *
from z3verifier import DNF_to_z3expr
from dnfs_and_transitions import list3D_to_listof2Darrays, dnfdisjunction, dnfTrue, dnfconjunction, dnfnegation, dnfFalse, transition
import itertools
from selection_points import v_representation, Dstate
import sympy

def isAffinePredicateInductive ( p, T  ):
    # p is a list, ptf is a 2d numpy array
    def isAffinePredicateptFInductive ( p, ptf  ):
        n = len(p) - 2
        w = np.array(p[:-2] + [p[-1]])
        newCoeffList = []
        for i in range(n+1):
            newCoeffList.append(  np.dot( ptf[: , i] , w)  )

        nonZeroIndex = next((i for i, x in enumerate(newCoeffList) if x), None)

        ratio = (w[nonZeroIndex] * 1.0) / newCoeffList[nonZeroIndex]

        newAffinePredicate = [ (x * ratio) for x in newCoeffList]

        for i in range(n+1):
            if (newAffinePredicate[i] != w[i]):
                return False

        return True

    for TransRel in T:
        for ptf in TransRel.tlist:
            if (not isAffinePredicateptFInductive(p, ptf)):
                return False

    return True

# S is list of 2d numpy arrays
def getAffinePredicates (S):
    def getCCAffinePredicates (cc_list):
        rv = []
        for p in cc:
            if (p[-2] == 0):
                rv.append(p.tolist())
                continue
            else: # Write code to take into account equality represented as different inequalities
                q = p.copy()
                q[-2] = 0
                rv.append(q.tolist())
        return rv
    
    rv = []
    for cc in S:
        rv = rv + getCCAffinePredicates(cc.tolist())
    
    return rv


def checkImplies( A, B ):
    n = len(A[0][0]) - 2
    def __get_cex(C):
        result = []
        s = Solver()
        s.add(Not(C))
        while len(result) < 1 and s.check() == sat: 
            m = s.model()
            result.append(m)
            # Create a new constraint that blocks the current model
            block = []
            for d in m:
                # d is a declaration
                if d.arity() > 0:
                    raise Z3Exception("uninterpreted functions are not supported")
                # create a constant from declaration
                c = d()
                if is_array(c) or c.sort().kind() == Z3_UNINTERPRETED_SORT:
                    raise Z3Exception("arrays and uninterpreted sorts are not supported")
                block.append(c != m[d])
            s.add(Or(block))
        else:
            if len(result) < 1 and s.check() != unsat: 
                print("Solver can't verify or disprove")
                return result
        return result

    return ( len(__get_cex(Implies( DNF_to_z3expr( A , primed = 0) , DNF_to_z3expr (B , primed = 0) )) ) == 0 )

# S is list of 2d numpy arrays, a is 1d numpy array or 1d list
def doesAffinePredicateSatisfySet (a , S):
    a_DNF = [np.array(a , ndmin = 2)]
    return checkImplies(S, a_DNF)

#P,Q are lists of 2d numpy arrays, T is a transition function
def modifiedHoudini( P, Q, T ):
    n = len(P[0][0]) - 2
    temp = []
    temp = temp + getAffinePredicates(P)
    temp = temp + getAffinePredicates(Q)


    rv = []

    for p in temp:
        flag = 0
        for a in rv:
            if (a == p or (a == [-x for x in p]) ):
                flag = 1
                break
        if (flag == 1):
            continue        
        if (doesAffinePredicateSatisfySet(p, P)  ): 
            if (isAffinePredicateInductive(p, T)):
                rv.append(p)


    if (len(rv) != 0):
        return [np.array(rv, ndmin = 2)]
    
    return []

def getIterativeP(P, B):
    A = dnfconjunction(P, B, 1)
    return P.copy() if (checkImplies(P, B)) else A.copy()

def getnonIterativeP(P, B, n):
    A = dnfconjunction(P, dnfnegation(B), 1)
    return [] if (checkImplies(P, B)) else A.copy()    



# def fullaffineSpace(n):
#     V = [ [0]*n ]
#     for i in range(n):
#         temp = [0] * n
#         temp[i] = 1
#         V.append(temp)
#     return V

# def affineHull (V1, V2):
#     V = V1.copy() + V2.copy()
#     mat = np.array(V)
#     _, inds = sympy.Matrix(mat).T.rref() 
#     return mat[inds].tolist()


# # Assumes P is a list of 2D numpy array with normalized predicates
# def AffineHullPrecondition (P):
    
#     n = len(P[0][0]) - 2
#     P_LII_in_Dstate = dnfconjunction( P , Dstate(n), 0)
    
#     rv = []
#     for cc in P_LII_in_Dstate:
#         rv = affineHull( rv,   v_representation(cc) )

#     return rv

# # ptf is 2d numpy array, P is list of 2d numpy arrays
# def KarrAnalysisSingleptf (P, ptf):
#     rv = AffineHull_Precondition(P)
#     for i in range(n):
#         rv = affineHull(rv, [ transition(x, ptf) for x in rv ] )

#     return rv 


# #TO DO: Convert to H representation
# def VtoH_affinespace (cc):
#     return


# def KarrAnalysis (P, T):

#     rv = []
#     for tr in T:
#         for ptf in tr.tlist:
#             rv = affineHull(rv, KarrAnalysisSingleptf(P, ptf) )


#     return rv 

# print(  affineHull (fullaffineSpace(3) , fullaffineSpace(3)) )