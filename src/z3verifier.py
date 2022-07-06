from z3 import *
import numpy as np
from dnfs_and_transitions import dnfnegation, dnfconjunction, dnfTrue
from configure import Configure as conf

def DNF_to_z3expr(I, primed):
    p = 'p' if primed else ''
    if np.size(I) == 0:
        return True

    d = len(I)
    c = len(I[0])
    n = len(I[0][0]) - 2
    return simplify(  Or([ And([ conf.OP[int(I[i][j][-2])](Sum([I[i][j][k] * Int(('x%s'+p) % k) 
        for k in range(n)]), int(I[i][j][-1])) for j in range(c) ]) for i in range(d) ]))


def genTransitionRel_to_z3expr(T):
    def ptf_to_z3expr(ptf):
        d = len(ptf)
        return simplify(And([Int("x%sp" % i) == Sum([ int(ptf[i][j]) * Int("x%s" % j) for j in 
            range(d-1) ]) + int(ptf[i][d-1]) for i in range(d-1) ]))

    def ptfp_to_z3expr(ptfp):
        return simplify(If(DNF_to_z3expr(ptfp.b, primed = 0), ptf_to_z3expr(ptfp.t), False))

    def ptfplist_to_z3expr(ptfplist):
        ret = False
        for ptfp in ptfplist:
            ret = simplify(And(ret, ptfp_to_z3expr(ptfp) ))
        return ret    

    ptfplist = T[0] + T[1]
    return ptfplist_to_z3expr(ptfplist)


def z3_verifier(P_z3, B_z3, T_z3, Q_z3, I_z3):
    def __get_cex(C):
        result = []
        s = Solver()
        s.add(Not(C))
        while len(result) < conf.s and s.check() == sat: 
            m = s.model()
            result.append(m)
            # Create a new constraint the blocks the current model
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
            if len(result) < conf.s and s.check() != unsat: 
                print("Solver can't verify or disprove")
                return result
        return result

    #P -> I
    def __get_cex_plus(P_z3, I_z3):
        return __get_cex(Implies(P_z3, I_z3))

    #B & I & T => I'
    def __get_cex_ICE(B_z3, I_z3, T_z3, Ip_z3):
        return __get_cex(Implies(And(B_z3, I_z3, T_z3), Ip_z3))

    # I -> Q
    def __get_cex_minus(I_z3, Q_z3):
        return __get_cex(Implies(And(I_z3, Q_z3)))

    cex_plus = __get_cex_plus(P_z3, I_z3)
    cex_minus = __get_cex_minus(I_z3, Q_z3)
    cex_ICE = __get_cex_ICE(B_z3, I_z3, T_z3, Ip_z3)
    correct = 1 if (len(cex_plus) + len(cex_minus) + len(cex_ICE) == 0) else 1
    return ( correct , (cex_plus, cex_minus, cex_ICE) )

    

P = [np.array([[1, 0, 0]])]
B = [np.array([[1, -1, 6]])]
Q = [np.array([[1, 0, 6]])]

class partialTransitionFuncPair:
    def __init__(self, transition_matrix, DNF, B):
        self.t = transition_matrix
        self.b = dnfconjunction(DNF, B, gLII = 1)

def detTransitionFunc(B, *args):
    return [partialTransitionFuncPair(x[0], x[1], B) for x in args]

def nondetTransitionRel(B, *args):
    return [partialTransitionFuncPair(x[0], x[1], B) for x in args]

def genTransitionRel(Dtf, Ntr):
    return [Dtf, Ntr]

T = genTransitionRel( detTransitionFunc( B,  [ np.array([[1, 1], [0, 1]]) , dnfTrue(1) ] ,  [ np.array([[1, 2], [0, 1]]) , dnfTrue(1) ]  ) , nondetTransitionRel(B ))
print(T)


T_z3 = genTransitionRel_to_z3expr(T)
print(T_z3)



