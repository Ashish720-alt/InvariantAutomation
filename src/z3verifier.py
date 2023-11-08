from z3 import *
import numpy as np
from configure import Configure as conf
from dnfs_and_transitions import dnfconjunction, dnfnegation, transition, deepcopy_DNF
from selection_points import Dstate, removeduplicatesICEpair, removeduplicates, v_representation, pointsatisfiescc
from math import floor, ceil, sqrt

def DNF_to_z3expr(I, primed):
    p = 'p' if primed else ''
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
        
        print("1") #Debug
        
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
                block.append(c != m[d])    
     
            s.add(Or(block))
        else:
            print("2") #Debug
            if len(result) < conf.s and s.check() != unsat: 
                print("2.1") #Debug
                print("Solver can't verify or disprove")
                return result
            print("3") #Debug
        
        return result

    # t > 0 => enlarge ; t < 0 => shrink
    def __get_enlargedI(I, t):
        I_copy = deepcopy_DNF(I)
        for cc in I_copy:
            for p in cc:
                magnitude = sqrt(sum(i*i for i in p[:-2]))
                p[-1] = p[-1] + floor( magnitude * t )
        return I_copy

    #P -> I
    def __get_cex_plus(P_z3, I, n):
        for t in conf.z3_C1_Tmax:
            I_enlarge = dnfconjunction(__get_enlargedI(I, t) , Dstate(n), 1)
            I_z3 = DNF_to_z3expr( I_enlarge, primed = 0)
            plus = convert_cexlist(__get_cex(Implies(P_z3, I_z3)), 0, n)
            if (len(plus) > 0):
                return plus
        return []

    #B & I & T => I'
    def __get_cex_ICE(B_z3, I, T_z3, n):
        def __get_cex_ICE_givenI(B_z3, T_z3, n, I_z3, Ip_z3):
            return convert_cexlist(__get_cex(Implies(And(B_z3, I_z3, T_z3), Ip_z3)), 1, n) 
            
    
        rv = []
        for t in conf.z3_C2_Tmax:       
            I_enlarge1 = dnfconjunction(__get_enlargedI(I, -t) , Dstate(n), 1)
            I_enlarge2 = dnfconjunction(__get_enlargedI(I, t) , Dstate(n), 1)
            I_z3 = DNF_to_z3expr( I_enlarge1, primed = 0)
            Ip_z3 = DNF_to_z3expr(I_enlarge2, primed = 1)                  
            rv = __get_cex_ICE_givenI(B_z3, T_z3, n, I_z3, Ip_z3)
            if (rv != []):
                return rv
        
        return []


    # I -> Q
    def __get_cex_minus(I, Q_z3, n):
        for t in conf.z3_C3_Tmax:
            I_enlarge = dnfconjunction(__get_enlargedI(I, -t), Dstate(n), 1) 
            I_z3 = DNF_to_z3expr( I_enlarge, primed = 0)
            minus = convert_cexlist(__get_cex(Implies(I_z3, Q_z3)), 0, n) 
            if (len(minus) > 0):
                return minus
        return []
    
    n = len(I[0][0]) - 2
    (cex_plus, cex_minus, cex_ICE) = ( __get_cex_plus(P_z3, I, n) ,__get_cex_minus(I, Q_z3, n) ,__get_cex_ICE(B_z3, I, T_z3, n))
    correct = 1 if (len(cex_plus) + len(cex_minus) + len(cex_ICE) == 0) else 0
    return ( correct , ( removeduplicates(cex_plus), removeduplicates(cex_minus), removeduplicatesICEpair(cex_ICE) ) )



# def centroids(dnf):
#     def centroid(cc):
#         v_repr = v_representation(cc)
#         if (v_repr == []):
#             return []
#         points = len(v_repr)
#         n = len(v_repr)  
#         coordinate_wise_sum = [0] * n
#         for pt in v_repr:
#             for i in range(n):
#                 coordinate_wise_sum[i] += pt[i]
#         centroid = [1.0 * x / points for x in coordinate_wise_sum]
#         return centroid # Centroids need not be lattice point, need nearest lattice point satisfying cc, if it exists
#     rv = []
#     for cc in dnf:
#         ctr = centroid(cc)
#         if (ctr != []):
#             rv.append(ctr)
#     return rv



# def ILP_verifier (P, B, T, Q, I):
#     def __get_cex_plus(P, I):
#         return centroids( dnfconjunction(P, dnfnegation(I), 0) ) # Centroids need not be lattice point, need nearest lattice point

#     def __get_cex_minus(I, Q):
#         return centroids( dnfconjunction(I, dnfnegation(Q), 0) )

#     def __get_cex_ICE(B, I, T):
#         #Assumes dnf in LII form
#         def Tinverse (dnf, ptf):
#             #Assumes cc in LII form; new equation is of the form A' x <= 0 where x is (n+1)*1 vector
#             def transform_cc(L):
#                 L_prime = []
#                 for row in L:
#                     new_row = row[:len(row)-2] + [row[-1] * -1]
#                     L_prime.append(new_row)
#                 return L_prime

#             #Assumes cc in LII form
#             def reverse_transform_cc(L_prime):
#                 cc = []
#                 for row in L_prime:
#                     new_row = row[:len(row)-1] + [-1,row[-1] * -1]
#                     cc.append(new_row)
#                 return cc


#             def Tinverse__cc(cc, ptf):
#                 cc_prime = transform_cc(cc)
#                 return reverse_transform_cc( np.matmul( np.array(cc_prime) , ptf ).tolist())

#             rv = []
#             for cc in dnf:
#                 rv.append(Tinverse__cc(cc, ptf))
#             return rv

#         rv = []
#         for rtf in T:
#             for ptf in rtf.tlist:
#                 heads = centroids( dnfconjunction(I, dnfconjunction(B , Tinverse(dnfnegation(I), ptf) , 1), 0) ) 
#                 for hd in heads:
#                     tl = transition(hd, ptf)
#                     rv.append((hd,tl))
#         return rv

#     n = len(I[0][0]) - 2
#     I_bounded = dnfconjunction(I, Dstate(n), 1)
#     (cex_plus, cex_minus, cex_ICE) = ( __get_cex_plus(P, I) ,__get_cex_minus(I, Q) ,__get_cex_ICE(B, I, T))
#     correct = 1 if (len(cex_plus) + len(cex_minus) + len(cex_ICE) == 0) else 0
#     return ( correct , ( removeduplicates(cex_plus), removeduplicates(cex_minus), removeduplicatesICEpair(cex_ICE) ) )    
    
    
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