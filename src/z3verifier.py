from z3 import *
import numpy as np
from configure import Configure as conf
from dnfs_and_transitions import dnfconjunction
from selection_points import Dstate, removeduplicatesICEpair, removeduplicates

def DNF_to_z3expr(I, primed):
    p = 'p' if primed else ''
    # FIXME: I don't know if we can do this. Is possible to have np.size(I[0]) == 0 but np.size(I) != 0?
    #        But np.size(I) == 0 will throw an error if I = [array(...), array(...), ...]
    if len(I) == 0 or np.size(I[0]) == 0:
        return True

    d = len(I)
    # c = len(I[0])
    n = len(I[0][0]) - 2
    return simplify(  Or([ And([ conf.OP[int(I[i][j][-2])](Sum([I[i][j][k] * Int(('x%s'+p) % k) 
        for k in range(n)]), int(I[i][j][-1])) for j in range(len(I[i])) ]) for i in range(d) ]))


def genTransitionRel_to_z3expr(T):
    def ptf_to_z3expr(ptf):
        n = len(ptf) - 1
        return simplify(And(And([Int("x%sp" % i) == Sum([ int(ptf[i][j]) * Int("x%s" % j) for j in 
            range(n) ]) + int(ptf[i][n]) for i in range(n) ]) , DNF_to_z3expr( Dstate(n) , primed = 1) ))

    def Btr_to_z3expr(Btr):
        return Implies( DNF_to_z3expr(Btr.b, primed = 0) , simplify(Or([ptf_to_z3expr(ptf) for ptf in Btr.tlist])) )  
 
    return simplify(And([ Btr_to_z3expr(Btr) for Btr in T  ]))



def z3_verifier(P_z3, B_z3, T_z3, Q_z3, I):
    def convert_cexlist(cexlist, ICEpair, n):
        def convert_cex(cex, ICEpair, n):
            if (ICEpair):
                return ([cex.evaluate(Int("x%s" % i), model_completion=True).as_long() for i in range(n)], [cex.evaluate(Int("x%sp" % i), model_completion=True).as_long() for i in range(n)] )
            else: 
                return [cex.evaluate(Int("x%s" % i), model_completion=True).as_long() for i in range(n)]     
        return [convert_cex(cex, ICEpair, n) for cex in cexlist]

    def __get_cex(C):
        result = []
        s = Solver()
        s.add(Not(C))
        while len(result) < conf.s and s.check() == sat: 
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
                # block.append(c != m[d])
                block.append( Or(c - m[d] > 100 , c - m[d] < -100) ) #Change to use ILP solver?!?         
            s.add(Or(block))
        else:
            if len(result) < conf.s and s.check() != unsat: 
                print("Solver can't verify or disprove")
                return result
        return result

    #P -> I
    def __get_cex_plus(P_z3, I_z3, n):
        return convert_cexlist(__get_cex(Implies(P_z3, I_z3)), 0, n)

    #B & I & T => I'
    def __get_cex_ICE(B_z3, I_z3, T_z3, Ip_z3, n):
        A = __get_cex(Implies(And(B_z3, I_z3, T_z3), Ip_z3))
        return convert_cexlist(A, 1, n) 

    # I -> Q
    def __get_cex_minus(I_z3, Q_z3, n):
        return convert_cexlist(__get_cex(Implies(I_z3, Q_z3)), 0, n) 
    
    n = len(I[0][0]) - 2
    I_bounded = dnfconjunction(I, Dstate(n), 1)
    (I_z3, Ip_z3) = (DNF_to_z3expr( I_bounded, primed = 0), DNF_to_z3expr(I_bounded, primed = 1))
    (cex_plus, cex_minus, cex_ICE) = ( __get_cex_plus(P_z3, I_z3, n) ,__get_cex_minus(I_z3, Q_z3, n) ,__get_cex_ICE(B_z3, I_z3, T_z3, Ip_z3, n))
    correct = 1 if (len(cex_plus) + len(cex_minus) + len(cex_ICE) == 0) else 0
    return ( correct , ( removeduplicates(cex_plus), removeduplicates(cex_minus), removeduplicatesICEpair(cex_ICE) ) )

    
# Testing:
# from dnfs_and_transitions import dnfnegation, dnfconjunction, dnfdisjunction, dnfTrue
# P = [np.array([[1, 0, 0]])]
# B = [np.array([[1, -1, 5]])]
# Q = [np.array([[1, -1, 6]])]

# class B_LItransitionrel:
#     def __init__(self, transition_matrix_list, DNF, B):
#         self.tlist = transition_matrix_list
#         self.b = dnfconjunction(DNF, B, gLII = 1)

# def genLItransitionrel(B, *args):
#     return [B_LItransitionrel(x[0], x[1], B) for x in args ]

# T = genLItransitionrel(B, ( [np.array([[1, 1], [0, 1]])] , dnfTrue(1) ) ) 

# P_z3 = DNF_to_z3expr(P, 0)
# B_z3 = DNF_to_z3expr(B, 0)
# Q_z3 = DNF_to_z3expr( dnfdisjunction(Q, B, 1), 0)
# T_z3 = genTransitionRel_to_z3expr(T)

# # print(P_z3, B_z3, Q_z3)
# # print( T_z3)

# I = [np.array([[-1, -1, 8]])]
# print(z3_verifier(P_z3, B_z3, T_z3, Q_z3, I)[1])