""" Representation of the logical system to be solved.
"""
import numpy as np
from dnfs_and_transitions import dnfconjunction
from selection_points import get_plus0, get_minus0, get_ICE0
from domain import D_p
from z3verifier import genTransitionRel_to_z3expr, DNF_to_z3expr
from configure import Configure as conf
from coefficientgraph import getrotationgraph
from math import sqrt, log
from preprocessing import modifiedHoudini, getIterativeP, getnonIterativeP
from print import print_colorslist
'''
The general single loop clause system is:
P -> I
I /\ B /\ T -> I'
I -> Q

'''

class B_LItransitionrel:
    def __init__(self, transition_matrix_list, DNF, B):
        self.tlist = transition_matrix_list
        self.b = dnfconjunction(DNF, B, gLII = 1)

def genLItransitionrel(B, *args):
    return [B_LItransitionrel(x[0], x[1], B) for x in args ]

class Repr:
    def __init__(self, P, B, T, Q, Var, c, d):
         
        self.n = len(P[0][0]) - 2  # n+1 is op, n+2 is const
         
        self.P = getIterativeP(P, B)
        self.P_noniters = getnonIterativeP(P, B, self.n)
        self.B = B.copy()
        self.Q = Q.copy()
        self.T = T.copy()
        self.Var = Var.copy()
        self.affineSubspace = modifiedHoudini(self.P, self.Q, self.T)


        self.P_z3expr = DNF_to_z3expr(self.P, primed = 0)
        self.B_z3expr = DNF_to_z3expr(self.B, primed = 0)
        self.Q_z3expr = DNF_to_z3expr(self.Q, primed = 0)
        self.T_z3expr = genTransitionRel_to_z3expr(self.T)

        self.c = c
        self.d = d
        self.tmax = conf.maxSArun

        self.plus0 = get_plus0(self.P, 0.5, conf.probenet_success)
        self.minus0 = get_minus0(self.Q, 0.5, conf.probenet_success)
        self.ICE0 = get_ICE0(self.T, 0.5, conf.probenet_success)        
                
        self.Dp = D_p(self.P, self.B, self.T, self.Q)

        self.k0 = max(self.Dp[0]) if (max(self.Dp[0]) < 100) else 1
        self.k1 = conf.dspace_radius * self.n * self.k0

        self.rotationgraph = getrotationgraph(self.k0, self.n) 

        self.rotationgraphvertices = self.rotationgraph[0]
        self.rotationgraphedges = self.rotationgraph[1]

        self.colorslist = print_colorslist(conf.num_processes)

    def get_n(self):
        return self.n

    def get_P(self):
        return self.P

    def get_nonItersP(self):
        return self.P_noniters

    def get_B(self):
        return self.B

    def get_Q(self):
        return self.Q

    def get_T(self):
        return self.T

    def get_Var(self):
        return self.Var

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

    def get_k0(self):
        return self.k0

    def get_k1(self):
        return self.k1

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

    def get_coeffvertices(self):
        return self.rotationgraphvertices

    def get_coeffneighbors(self, coeff):
        return self.rotationgraphedges[tuple(coeff)]

    def get_affineSubspace(self):
        return self.affineSubspace

    def get_colorslist(self):
        return self.colorslist
