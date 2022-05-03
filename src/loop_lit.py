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
'''


class loop_lit:

    # Standard Loop, Mixed Guard, variable_vector = (x,y)
    afnp2014 = get_input(P=np.array([[[1, 0, 0, 1], [0, 1, 0, 0] ]]),
                B=np.array([[[0, 1, -2, 1000]]]),
                Q=np.array([[[1, -1, 1, 0]]]),
                T=repr.SimpleTotalTransitionFunc(np.array([[1, 1, 0], [0, 1, 1],  [0, 0, 1]]))) 

    # Standard Loop, Randomized Guard, variable_vector = (x,y)
    bhmr2007 = get_input(P=np.array([[[1, 0, 0, 1], [0, 1, 0, 0] ]]),
                B=np.array([[[0, 1, -2, 1000]]]),
                Q=np.array([[[1, -1, 1, 0]]]),
                T=repr.SimpleTotalTransitionFunc(np.array([[1, 1, 0], [0, 1, 1],  [0, 0, 1]]))) 