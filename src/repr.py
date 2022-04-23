""" Representation of the logical system to be solved.
"""
import numpy as np
from dnf import DNF_to_z3expr, trans_func_to_z3expr

class PartialTransitionFunc:
    def __init__(self, DNF, transition_matrix):
        self.b = DNF
        self.t = transition_matrix


def TotalTransitionFunc(A):
    return [PartialTransitionFunc(np.zeros((1,1,len(A[0]) + 1) ), A)] #len(A[0]) is n+1, reqd is n+2

class Repr:
    def __init__(self, P, B, Q, T):
        """ 
        TODO: T description.
        """
        self.num_var = len(P[0][0]) - 2 # n+1 is op, n+2 is const
        self.P = P.copy()
        self.P_z3expr = DNF_to_z3expr(P)
        self.B = B.copy()
        self.B_z3expr = DNF_to_z3expr(B)
        self.Q = Q.copy()
        self.Q_z3expr = DNF_to_z3expr(Q)
        self.T = TotalTransitionFunc(T)
        self.T_z3expr = trans_func_to_z3expr(self.T)

        def _extract_consts():
            f = lambda x: set(x.flatten())
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



