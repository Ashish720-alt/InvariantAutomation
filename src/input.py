# Testing
from dnfs_and_transitions import dnfTrue
import numpy as np
from repr import genLItransitionrel, Repr
from main import metropolisHastings


#IMP: Remember Q is Q \/ B

class handcrafted:
    class mock:
        P = [np.array([[1, 0, 0]])]
        B = [np.array([[1, -1, 5]])]
        Q = [np.array([[1, -1, 6]])] 
        T = genLItransitionrel(B, ( [np.array([[1, 1], [0, 1]])] , dnfTrue(1) ) ) 

class loop_lit:
    class afnp2014: #For this, why does the coefficient take > |2| values when I put c = 2?
        P = [np.array([[1, 0, 0, 1], [0, 1, 0, 0] ])]
        B = [np.array([[0, 1, -2, 1000]])]
        Q = [np.array([[1, -1, 1, 0]])]
        T = genLItransitionrel(B, ( [np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])] , dnfTrue(2) ) ) 


P = handcrafted.mock.P
B = handcrafted.mock.B
T = handcrafted.mock.T
Q = handcrafted.mock.Q
metropolisHastings(Repr(P, B, T, Q))