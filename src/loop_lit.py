import repr
from input import get_input

''' Clause Systems:

SL, Randomized Guard:
P -> I
I(s) /\ T(s,t) -> I(t)
I -> Q

SL, Mixed Guard:
P -> I
I(s) /\ B(s) /\ T(s,t) -> I(t)
I /\ B-> Q
I /\ ~B -> Q

Simplify final two to equivalent: I -> Q

TL_DSNV, double sequential nested variation:
P -> I1
I1(s) /\ B1(s) /\ T1(s,t) -> I2(t)
I2(s) /\ B2(s) /\ T2(s,t) -> I2(t)
I2(s) /\ ~B2(s) /\ T3(s,t) -> I3(t)
I3(s) /\ B3(s) /\ T4(s,t) -> I3(t)
I3(s) /\ ~B3(s) /\ T5(s,t) -> I1(t)
I1 /\ ~B1 -> Q

'''


class loop_lit:

    # Standard Loop, Mixed Guard, variable_vector = (x,y)
    afnp2014 = get_input(P=np.array([[[1, 0, 0, 1], [0, 1, 0, 0] ]]),
                B=np.array([[[0, 1, -2, 1000]]]),
                Q=np.array([[[1, -1, 1, 0]]]),
                T=repr.SimpleTotalTransitionFunc(np.array([[1, 1, 0], [0, 1, 1],  [0, 0, 1]]))) 

    ''' This one is incomplete'''
    # SSL, variable_vector = (i,n,a,b) , small-finite transition relation 
    bhmr2007 = get_input(P=np.array([[[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, -1, 1000000], ]]),
                B=np.array([[ [1, -1, 0, 0, -2, 0] ]]),
                Q=np.array([[[0, -3, 1, 1, 0, 0]]]),
                T=repr.SimpleTotalTransitionFunc(np.array([[1, 1, 0], [0, 1, 1],  [0, 0, 1]]))) # The transition function is still wrong!!

    # SSL, variable_vector = (i,j) 
    cggmp2005 = get_input(P=np.array([[[1, 0, 0, 1], [0, 1, 0, 10] ]]),
                B=np.array([[[-1, 1, 1, 0]]]),
                Q=np.array([[[0, 1, 0, 6]]]),
                T=repr.SimpleTotalTransitionFunc(np.array([[1, 0, 2], [0, 1, -1],  [0, 0, 1]])))  

    # SSL, variable_vector = (low,mid, high) 
    cggmp2005_variant = get_input(P=np.array([[[1, 0, 0, 0, 0], [0, -2, 1, 0, 0] , [0, 1, 0, 2, 0] , [0, 1, 0, -1, 1000000]  ]]),
                B=np.array([[[0, 1, 0, 2, 0]]]),
                Q=np.array([[[1, 0, -1, 0 , 0]]]),
                T=repr.SimpleTotalTransitionFunc(np.array([[1, 0, 0, 1], [0, 1, 0, -1],  [0, 0, 1, -1], [0, 0, 0, 1] ])))  

    # TL_DSNV, variable_vector = (i,j,k) 
    cggmp2005b = get_input(P=np.array([[[1, 0, 0, 0, 0], [0, 1, 0, 0, -100] , [0, 0, 1, 0, 9]   ]]),
                B1=np.array([[[1, 0, 0, -1, 100]]]),
                B2=np.array([[[0, 1, 0, -2, 20]]]),
                B3=np.array([[[0, 0, 1, -1, 3]]]),
                Q=np.array([[[0, 0, 1, 0 , 4]]]),
                T1=repr.SimpleTotalTransitionFunc(np.array([[1, 0, 0, 1], [0, 1, 0, 0],  [0, 0, 1, 0], [0, 0, 0, 1] ])),
                T2=repr.SimpleTotalTransitionFunc(np.array([[1, 0, 0, 0], [1, 1, 0, 0],  [0, 0, 1, 0], [0, 0, 0, 1] ])),
                T3=repr.SimpleTotalTransitionFunc(np.array([[1, 0, 0, 0], [0, 1, 0, 0],  [0, 0, 0, 4], [0, 0, 0, 1] ])),
                T4=repr.SimpleTotalTransitionFunc(np.array([[1, 0, 0, 0], [0, 1, 0, 0],  [0, 0, 1, 1], [0, 0, 0, 1] ])),
                T5=repr.SimpleTotalTransitionFunc(np.array([[1, 0, 0, 0], [0, 1, 0, 0],  [0, 0, 1, 0], [0, 0, 0, 1] ])))  