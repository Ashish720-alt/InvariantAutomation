import repr
import numpy as np
import cdd
from configure import Configure as conf
from dnf import dnfconjunction, dnfnegation, dnfdisjunction, dnfTrue, dnfFalse, genLII_to_LII, transition


def Dstate(n):
    cc = np.empty(shape=(0,n+2), dtype = int)
    for i in range(n):
        p1 = np.zeros(n+2)
        p1[n] = -1
        p1[i] = -1
        p1[n+1] = -1 * conf.intmin

        p2 = np.zeros(n+2)
        p2[n] = -1
        p2[i] = 1
        p2[n+1] = conf.intmax
        
        cc = np.concatenate((cc, np.array([p1, p2], ndmin=2)))
    
    return cc
        

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


def get_plus0 (P, n): 
    P_LII_in_Dstate = dnfconjunction( P , Dstate(n), 0)

    plus0 = []
    for cc in P_LII_in_Dstate:
        plus0 = plus0 + v_representation(cc)

    return plus0


def get_minus0 (Q, n):
    negQ_LII_in_Dstate = dnfconjunction( dnfnegation(Q) , Dstate(n), 0) 

    minus0 = []
    for cc in negQ_LII_in_Dstate:
        minus0 = minus0 + v_representation(cc)

    return minus0

# An ICE pair is a list of 2 states, where a state is a list of n numbers
def get_ICE0 (T, B, n):
    alltransfuncpairs = T[0] + T[1]

    ICE0 = conf.X_ICE
    for transfuncpair in alltransfuncpairs:
        B = dnfconjunction( transfuncpair.b, Dstate(n) , 0)
        T = transfuncpair.t
        heads_of_ICEpairs = []
        for cc in B: 
            heads_of_ICEpairs = heads_of_ICEpairs + v_representation(cc)
        for head in heads_of_ICEpairs:
            tail = transition(head, T)
        ICE0.append([head, tail])
    
    return ICE0



#Testing:
# This cc represents y >= 0, x <= 1, y <= x.
# print(v_representation( np.array([[0,-1,-1,0] , [1,0, -1, 1], [-1, 1, -1, 0] ]) ))
# print(Dstate(2))