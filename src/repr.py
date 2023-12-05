""" Representation of the logical system to be solved.
"""
import numpy as np
from dnfs_and_transitions import dnfconjunction
from selection_points import Dstate, get_plus0, get_minus0, get_ICE0, TIterates
from domain import D_p
from z3verifier import genTransitionRel_to_z3expr, DNF_to_z3expr
from configure import Configure as conf
from coefficientgraph import getrotationgraph
from math import sqrt, log
from preprocessing import modifiedHoudini, getIterativeP, getnonIterativeP
from print import print_colorslist
from enetspointcounting import getpoints
'''
The general single loop clause system is:
P -> I
I /\ B /\ T -> I'
I -> Q

'''

class B_LItransitionrel:
    def __init__(self, transition_matrix_list, DNF, B):
        self.tlist = transition_matrix_list
        self.b = dnfconjunction(DNF, B, gLII = 0)

def genLItransitionrel(B, *args):
    return [B_LItransitionrel(x[0], x[1], B) for x in args ]

class Repr:
    def __init__(self, P, B, T, Q, Var, c, d):
        
        self.n = len(P[0][0]) - 2  # n+1 is op, n+2 is const
         
        if (conf.NONITERATIVE_PRECONDITION == conf.ON):
            self.P = getIterativeP(P, B)
            self.P_noniters = getnonIterativeP(P, B, self.n)
        else:
            self.P = P
            self.P_noniters = []         
        self.B = B.copy()
        self.Q = Q.copy()
        self.T = T.copy()
        self.Var = Var.copy()
        
        if (conf.AFFINE_SPACES == conf.ON):
            self.affineSubspace = modifiedHoudini(self.P, self.Q, self.T)
        else:
            self.affineSubspace = []


        self.P_z3expr = DNF_to_z3expr( dnfconjunction(self.P, Dstate(self.n), 1), primed = 0)
        self.B_z3expr = DNF_to_z3expr(dnfconjunction(self.B, Dstate(self.n), 1), primed = 0)
        self.Q_z3expr = DNF_to_z3expr(dnfconjunction(self.Q, Dstate(self.n), 1), primed = 0)
        self.T_z3expr = genTransitionRel_to_z3expr(self.T)

        self.c = c
        self.d = d
        self.tmax = conf.maxSArun

        self.curr_enet_Size = getpoints(self.n, conf.e0, conf.probenet_success, 0)[0]
        
        self.transitionIterates = TIterates(self.T)
        
        self.plus0 = get_plus0(self.P, self.curr_enet_Size)
        self.minus0 = get_minus0(self.Q, self.curr_enet_Size) 
        self.ICE0 = get_ICE0(self.T, self.P, self.Q, self.curr_enet_Size, self.transitionIterates)        

        
        self.Dp = D_p(self.P, self.B, self.T, self.Q)

        self.k0 = max(2, max(self.Dp[0])) if (max(self.Dp[0]) < 100) else 2
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

    def get_coeffedges(self):
        return self.rotationgraphedges

    def get_affineSubspace(self):
        return self.affineSubspace

    def get_colorslist(self):
        return self.colorslist
    
    def update_enet(self, e, samplepoints):
        required_enetSize = getpoints(self.n, e, conf.probenet_success, self.curr_enet_Size)[0]
        (plus, minus, Implpair) = (samplepoints[0], samplepoints[1], samplepoints[2])
        plus = get_plus0(self.P, required_enetSize - self.curr_enet_Size)
        minus = get_minus0(self.Q, required_enetSize - self.curr_enet_Size)
        Implpair = get_ICE0(self.T, self.P, self.Q, required_enetSize - self.curr_enet_Size,  self.transitionIterates)  
        self.curr_enet_Size = required_enetSize           
        return (plus, minus, Implpair)
    
    def get_transitionIterates(self):
        return self.transitionIterates