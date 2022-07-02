""" Representation of the logical system to be solved.
"""
import numpy as np
from dnfs_and_transitions import DNF_to_z3expr, trans_func_to_z3expr, dnfconjunction

'''
The general clause system is:
P -> I
I /\ B /\ T -> I'
I -> Q

'''

# A partialTransitionFuncPair is a pair(t_i, B_i /\ B) where T is a partial LI transition function, and B_i, B are dnfs
class partialTransitionFuncPair:
    def __init__(self, transition_matrix, DNF, B):
        self.t = transition_matrix
        self.b = dnfconjunction(DNF, B, 1)

def detTransitionFunc(*args, B):
    return [partialTransitionFuncPair(x[0], x[1], B) for x in args]

def nondetTransitionRel(*args, B):
    return [partialTransitionFuncPair(x[0], x[1], B) for x in args]

def genTransitionRel(Dtf, Ntr):
    return [Dtf, Ntr]

class Repr:
    def __init__(self, P, B, T, Q):

        self.n = len(P[0][0]) - 2  # n+1 is op, n+2 is const
        
        self.P = P.copy()
        self.B = B.copy()
        self.Q = Q.copy()
        self.T = T.copy()
        
        self.P_z3expr = DNF_to_z3expr(P)
        self.B_z3expr = DNF_to_z3expr(B)
        self.Q_z3expr = DNF_to_z3expr(Q)
        self.T_z3expr = trans_func_to_z3expr(self.T)

        def _extract_pc():
            def f(dnf): 
                ret = set()
                for cc in dnf: 
                    ret = ret | set(cc.flatten()) 
                return ret

            ret = f(self.P) | f(self.B) | f(self.Q)
            ptfp_list = self.T[0] + self.T[1] 
            for ptfp in ptfp_list:
                ret |= f(ptfp.b) |  set(ptfp.t.flatten()) 
            return list(ret) 
        
        self.pc = _extract_pc()

    def get_n(self):
        return self.n

    def get_pc(self):
        return self.pc

    def get_P(self):
        return self.P

    def get_B(self):
        return self.B

    def get_Q(self):
        return self.Q

    def get_T(self):
        return self.T

    def get_P_z3expr(self):
        return self.P_z3expr

    def get_B_z3expr(self):
        return self.B_z3expr

    def get_Q_z3expr(self):
        return self.Q_z3expr

    def get_T_z3expr(self):
        return self.T_z3expr






def I(n):
    return np.identity(n + 1, dtype=int)

def E(n, pos ): #Indices run from 1 to n+1
    T = np.zeros(shape=(n+1, n+1), dtype=int)
    T[pos[0]-1][pos[1]-1] = 1
    return T



class PartialTransitionFunc:
    def __init__(self, DNF, transition_matrix):
        self.b = DNF
        self.t = transition_matrix


def TotalTransitionFunc(*args):
    return [PartialTransitionFunc(x[0], x[1]) for x in args]


def SimpleTotalTransitionFunc(A):
    # len(A[0]) is n+1, reqd is n+2
    return [ PartialTransitionFunc(np.zeros((1, 1, len(A[0]) + 1)), A) ]



class Repr:
    def __init__(self, P, B, Q, T):
        """ 
        TODO: T description.
        """
        self.num_var = len(P[0][0]) - 2  # n+1 is op, n+2 is const
        self.P = P.copy()
        self.P_z3expr = DNF_to_z3expr(P)
        self.B = B.copy()
        self.B_z3expr = DNF_to_z3expr(B)
        self.Q = Q.copy()
        self.Q_z3expr = DNF_to_z3expr(Q)
        self.T = T.copy()
        self.T_z3expr = trans_func_to_z3expr(self.T)

        def _extract_consts():
            def f(x): return set(x.flatten())
            ret = f(self.P) | f(self.B) | f(self.Q)
            for partial in self.T:
                ret |= f(partial.b) | f(partial.t)
            return list(ret) 
        self.consts = _extract_consts()

    def get_num_var(self):
        return self.num_var

    def get_consts(self):
        return self.consts

    def get_P(self):
        return self.P

    def get_B(self):
        return self.B

    def get_Q(self):
        return self.Q

    def get_T(self):
        return self.T

    def get_P_z3expr(self):
        return self.P_z3expr

    def get_B_z3expr(self):
        return self.B_z3expr

    def get_Q_z3expr(self):
        return self.Q_z3expr

    def get_T_z3expr(self):
        return self.T_z3expr
