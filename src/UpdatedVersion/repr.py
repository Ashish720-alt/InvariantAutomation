""" Representation of the logical system to be solved.
"""
import numpy as np
from dnf import DNF_to_z3expr, DNF_to_z3expr_p, trans_func_to_z3expr

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
    def __init__(self, CHC, DNFs, trans_funcs, invs):
        self.CHC = CHC
        
        self.num_var = len(DNFs[CHC[0][0][0]]) - 2 # n+1 is op, n+2 is const
        
        self.DNFs = DNFs
        self.trans_funcs = trans_funcs

        self.DNFs_z3expr = {}
        for dnf in self.DNFs:
            self.DNFs_z3expr[dnf] = DNF_to_z3expr(self.DNFs[dnf])

        self.DNFs_p_z3expr = {}
        for dnf in self.DNFs:
            self.DNFs_p_z3expr[dnf] = DNF_to_z3expr_p(self.DNFs[dnf])

        self.trans_funcs_z3expr = {}
        for trans_func in self.trans_funcs:
            self.trans_funcs_z3expr[trans_func] = trans_func_to_z3expr(self.trans_funcs[trans_func])

        def _extract_consts():
            def f(x): return set(x.flatten())
            ret = set()
            for dnf in self.DNFs:
                ret |= f(self.DNFs[dnf])
            for trans_func in self.trans_funcs:
                for partial in trans_func:
                    ret |= f(partial.b) | f(partial.t)
            return list(ret) 
        self.consts = _extract_consts()

        self.invs = invs

    def get_num_var(self):
        return self.num_var

    def get_consts(self):
        return self.consts

    def get_DNF(self, name):
        return self.DNFs[name]

    def get_trans_func(self, name):
        return self.trans_funcs[name]

    def get_DNF_z3expr(self, name):
        return self.DNFs_z3expr[name]

    def get_trans_func_z3expr(self, name):
        return self.trans_funcs_z3expr[name]

