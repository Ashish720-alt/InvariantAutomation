""" Representation of the logical system to be solved.
"""
import numpy as np
from dnfs_and_transitions import dnfconjunction
import selection_points
from domain import D_p
from z3_verifier import genTransitionRel_to_z3expr, DNF_to_z3expr

'''
The general single loop clause system is:
P -> I
I /\ B /\ T -> I'
I -> Q

'''

# A partialTransitionFuncPair is a pair(t_i, B_i /\ B) where T is a partial LI transition function, and B_i, B are dnfs
class partialTransitionFuncPair:
    def __init__(self, transition_matrix, DNF, B):
        self.t = transition_matrix
        self.b = dnfconjunction(DNF, B, gLII = 1)

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
        self.T_z3expr = genTransitionRel_to_z3expr(self.T)

        self.c = 3
        self.d = 3
        self.tmax = 10000
        self.X_ICE = []

        self.plus0 = get_plus0(P)
        self.minus0 = get_minus0(Q)
        self.ICE0 = get_ICE0(T, self.X_ICE)        
                
        self.Dp = D_p(self.P, self.B, self.T, self.Q)

    def get_n(self):
        return self.n

    def get_P(self):
        return self.P

    def get_B(self):
        return self.B

    def get_Q(self):
        return self.Q

    def get_T(self):
        return self.T

    def get_plus0(self):
        return self.plus0

    def get_minus0(self):
        return self.minus0    

    def get_ICE0(self):
        return self.ICE0   

    def get_Dp(self):
        return self.Dp 

    def get_c(self):
        return self.c

    def get_d(self):
        return self.d

    def get_tmax(self):
        return self.tmax

    def get_P_z3expr(self):
        return self.P_z3expr

    def get_B_z3expr(self):
        return self.B_z3expr

    def get_Q_z3expr(self):
        return self.Q_z3expr

    def get_T_z3expr(self):
        return self.T_z3expr




