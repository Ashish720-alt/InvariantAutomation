import numpy as np
from z3 import *
from z3verifier import DNF_to_z3expr
from dnfs_and_transitions import list3D_to_listof2Darrays, dnfdisjunction, dnfTrue, dnfconjunction




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

# S is list of 2d numpy arrays, a is 1d numpy array or 1d list
def doesAffinePredicateSatisfySet (a , S):
    #a -> S
    a_DNF = [np.array(a , ndmin = 2)]
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

    def isTrue(a_DNF, S, n):
        return (len(__get_cex(Implies( DNF_to_z3expr( S , primed = 0) , DNF_to_z3expr (a_DNF , primed = 0) )) ) == 0)

    return isTrue(a_DNF, S, len(a) - 2)

#P,Q are lists of 2d numpy arrays, T is a transition function
def affineSubspace( P, Q, T ):
    n = len(P[0][0]) - 2
    temp = []
    temp = temp + getAffinePredicates(P)
    temp = temp + getAffinePredicates(Q)


    rv = []

    for p in temp:
        flag = 0
        for a in rv:
            if (a == p):
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


# class B_LItransitionrel:
#     def __init__(self, transition_matrix_list, DNF, B):
#         self.tlist = transition_matrix_list
#         self.b = dnfconjunction(DNF, B, gLII = 1)

# def genLItransitionrel(B, *args):
#     return [B_LItransitionrel(x[0], x[1], B) for x in args ]

# A = np.array([[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]] )
# s = [1, -1, 1, 0, 0]
# P = [ np.array( [[1,0,1,1,0], [4,3,2,-1,0], [4,3,2,0,0], [1, -1, 1, 0, 0]] ) , np.array( [[1,0,1,0,0], [4,3,2,-2,0], [4,3,2,0,0] , [1, -1, 1, 0, 0]] ) ]
# Q =  [ np.array( [[1,0,17,1,0], [4,37,2,-1,0], [4,7,2,0,0], [1, -1, 1, 0, 0]] ) , np.array( [[1,0,71,0,0], [4,3,72,-2,0], [4,3,27,0,0] , [1, -1, 1, 0, 0]] ) ]
# P = list3D_to_listof2Darrays([[[1, 0, -1, 0, 0], [0, 1, -1, 0, 0], [0, 0, 1, 2, 0]]])
# B = list3D_to_listof2Darrays([[[0, 0, 1, 2, 0], [1, 0, 0, 2, 0]]])
# Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 1, 0, 0, 0]]]) , B , 1)
# T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, 0], [0, 0, 0, 1]] ) ] , dnfTrue(3)) )
# # print(isAffinePredicateInductive(s, A))
# # print(getAffinePredicates(P))
# # print(doesAffinePredicateSatisfySet(s, P))
# print(affineSubspace(P,Q,T))