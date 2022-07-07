
import numpy as np
import cdd
from configure import Configure as conf
from dnfs_and_transitions import dnfconjunction, dnfnegation, dnfdisjunction, dnfTrue, dnfFalse, genLII_to_LII, transition


def Dstate(n):
    cc = np.empty(shape=(0,n+2), dtype = int)
    for i in range(n):
        p1 = np.zeros(n+2)
        p1[n] = -1
        p1[i] = -1
        p1[n+1] = -1 * conf.dspace_intmin

        p2 = np.zeros(n+2)
        p2[n] = -1
        p2[i] = 1
        p2[n+1] = conf.dspace_intmax
        
        cc = np.concatenate((cc, np.array([p1, p2], ndmin=2)))
    
    return [cc]
        

# Assumes cc has LI predicates only, and not genLI predicates
def v_representation (cc):
    def pred_to_matrixrow (p):
        matrixrow = np.roll(p * -1, 1)[:-1]
        matrixrow[0] = matrixrow[0] * -1
        return matrixrow

    mat = []
    for p in cc:
        mat.append(pred_to_matrixrow(p))
    
    mat_cdd = cdd.Matrix( mat, number_type='fraction')
    mat_cdd.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat_cdd)
    
    v_repr = poly.get_generators()
    tuple_of_tuple_generators = v_repr.__getitem__(slice(v_repr.row_size))
    list_of_list_generators = []
    for tuple_generator in tuple_of_tuple_generators:
        list_of_list_generators.append(list(tuple_generator)[1:])

    return list_of_list_generators


def get_plus0 (P): 
    n = len(P[0][0]) - 2
    P_LII_in_Dstate = dnfconjunction( P , Dstate(n), 0)

    plus0 = []
    for cc in P_LII_in_Dstate:
        plus0 = plus0 + v_representation(cc)

    return plus0


def get_minus0 (Q):
    n = len(Q[0][0]) - 2
    negQ_LII_in_Dstate = dnfconjunction( dnfnegation(Q) , Dstate(n), 0) 

    minus0 = []
    for cc in negQ_LII_in_Dstate:
        minus0 = minus0 + v_representation(cc)

    return minus0

# An ICE pair is a pair of 2 states, where a state is a list of n numbers
def get_ICE0 (T):
    def get_ICEfromBtr (Btr, n):
        B = dnfconjunction( Btr.b, Dstate(n) , 0)
        heads_of_ICEpairs = []
        for cc in B: 
            heads_of_ICEpairs = heads_of_ICEpairs + v_representation(cc)

        ret = []
        for head in heads_of_ICEpairs:
            for ptf in Btr.tlist:
                ret.append( (head, transition(head, ptf)) )
        return ret

    n = len( (T[0].b)[0][0] ) - 2
    X_ICE = [] # Need to write a function for this!
    ICE0 = X_ICE 
    for Btr in T:
        ICE0.append( get_ICEfromBtr(Btr, n) )
    return ICE0



# Testing:


# print(Dstate(2))

# # This cc represents y >= 0, x <= 1, y <= x.
# print(v_representation( np.array([[0,-1,-1,0] , [1,0, -1, 1], [-1, 1, -1, 0] ]) ))

# print(get_plus0([np.array([[0,-1,-1,0] , [1,0, -1, 1], [-1, 1, -1, 0] ])]  ))
# print(get_minus0([np.array([[0,-1,-1,0] , [1,0, -1, 1], [-1, 1, -1, 0] ])]  ))

# import repr

# B = [np.array([[1, -1, 5]])]

# class B_LItransitionrel:
#     def __init__(self, transition_matrix_list, DNF, B):
#         self.tlist = transition_matrix_list
#         self.b = dnfconjunction(DNF, B, gLII = 1)

# def genLItransitionrel(B, *args):
#     return [B_LItransitionrel(x[0], x[1], B) for x in args ]

# T = genLItransitionrel(B, ( [np.array([[1, 1], [0, 1]])] , dnfTrue(1) ) ) 

# print(get_ICE0(  T   )) 