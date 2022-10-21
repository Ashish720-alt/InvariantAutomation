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

    # Variable vector: (x,y)
    class c2d1_1:
        P = [np.array([[1, 0, 0, 1], [0, 1, 0, 1] ])]
        B = dnfTrue(2)
        Q = [np.array([[1, -1, 1, 0], [0,1,1,1] ])]
        T = genLItransitionrel(B, ( [np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])] , dnfTrue(2) ) )         

class loop_lit:
    # Variable vector: (x,y)
    class afnp2014:
        P = [np.array([[1, 0, 0, 2], [0, 1, 0, 2] ])]
        B = [np.array([[0, 1, -2, 1000]])]
        Q = [np.array([[1, -1, 1, 0]])]
        T = genLItransitionrel(B, ( [np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])] , dnfTrue(2) ) ) 

    # Variable vector: (a,b,i,n)
    class bhmr2007:
        P = [np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, -1, 10000] ])]
        B = [np.array([[0, 0, 1,-1, -1, 0]])]
        Q = [np.array([[1, 1, 0, -3, 0, 0]]), np.array([[0, 0, 1 ,-1, -1, 0]])]
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, 0, 1], [0, 1, 0, 0, 2], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]]) , 
                                       np.array([[1, 0, 0, 0, 2], [0, 1, 0, 0, 1], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]])  ] , dnfTrue(4) ) ) 
    
    # Variable vector: (i,j)
    class cggmp2005:
        P = [np.array([[1, 0, 0, 1], [0, 1, 0, 10] ])]
        B = [np.array([[-1, 1, 1, 0]])]
        Q = [np.array([[0, 1, 0, 6]]), np.array([[-1, 1, 1, 0]])]
        T = genLItransitionrel(B, ( [np.array([[1, 0, 2], [0, 1, -1], [0, 0, 1]])] , dnfTrue(2) ) )         

    # Variable vector: (x,y)
    class gsv2008:
        P = [np.array([[1, 0, 0, -50], [0, 1, 2, 1000], [0,1,-2,10000] ])]
        B = [np.array([[1, 0, -2, 0]])]
        Q = [np.array([[0, 1, 2, 0]]), np.array([[1, 0, -2, 0]])]
        T = genLItransitionrel(B, ( [np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])] , dnfTrue(2) ) )    

# This takes too much time.
class lineararbitrary:
    class testcase1:
        # variable vector = (h,i,j,k, m,n)
        P = [np.array([ [0,0,0,0,0,1,1,0] , [0,0,0,0,0,1,-1,200], [0,1,0,0,0,0,0,0], [0,0,0,0,1,0,0,0] , [0,0,1,0,0,1,0,0] , [0,0,0,1,0,1,0,0] , [1,0,0,0,0,1,0,0] ])]
        B = [np.array([[0,0,1,0,0,0,2,0] ])]
        Q = [np.array([[0,0,1,0,0,0,2,0], [0,1,0,0,0,0,1,0]])]
        T = genLItransitionrel(B, ( [np.array([[0,0,1,0,1,0, 0], [0,1,0,0,0,0, 0], [0,0,1,0,0,0,-1], [0,0,0,1,0,0,0], 
                        [0,0,0,0,1,0, 1], [0,0,0,0,0,1, 0] ])] , dnfTrue(6) ) ) 


P = loop_lit.cggmp2005.P
B = loop_lit.cggmp2005.B
T = loop_lit.cggmp2005.T
Q = loop_lit.cggmp2005.Q
metropolisHastings(Repr(P, B, T, Q))


''''''''''''''''''''''''''''''''''''''
# Hill Climbing Algorithm:
# from hillclimbing import hill_climbing
# hill_climbing(Repr(P, B, T, Q))