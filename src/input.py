# Testing
from dnfs_and_transitions import dnfTrue
import numpy as np
from repr import genLItransitionrel, Repr
from main import metropolisHastings


#IMP: Remember Q is Q \/ B for standard CHC

class handcrafted:
    class mock:
        P = [np.array([[1, 0, 0]])]
        B = [np.array([[1, -1, 5]])]
        Q = [np.array([[1, -1, 6]])] 
        T = genLItransitionrel(B, ( [np.array([[1, 1], [0, 1]])] , dnfTrue(1) ) ) 

class loop_lit:
    class afnp2014:
        P = [np.array([[1, 0, 0, 1], [0, 1, 0, 0] ])]
        B = [np.array([[0, 1, -2, 1000]])]
        Q = [np.array([[1, -1, 1, 0]])]
        T = genLItransitionrel(B, ( [np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])] , dnfTrue(2) ) ) 


P = loop_lit.afnp2014.P
B = loop_lit.afnp2014.B
T = loop_lit.afnp2014.T
Q = loop_lit.afnp2014.Q
metropolisHastings(Repr(P, B, T, Q))


''''''''''''''''''''''''''''''''''''''
# Hill Climbing Algorithm:
# from hillclimbing import hill_climbing
# hill_climbing(Repr(P, B, T, Q))