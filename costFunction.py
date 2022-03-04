import numpy as np 
from z3 import *

# Only considering the 1 variable case.
# n = 1 

'''Operator code:
g = 2
ge = 1
eq = 0
le = -1
l = -2 '''



''' A dnf (P, Q, B) which k conjuncts, each of which has j_i atomic predicates (1 <= i <= k) is a 3D array of k elements, with the (i-1)th element a 2D array with j_i element which each element of a 2D array a n + 2 dimensional 1D arary
(n coefficients, 1 operator value, 1 constant value)'''

''' Notation: The notation [1,0,2] represents the atomic formula over 1 predicate: x == 2'''

''' T will have 2n + 2 dimensional 1D arrays as atomic predicates. The notation for T is:


'''

# P = np.array([1, 0, 0] , ndmin = 3 )
# B = np.array([1, -2, 5] , ndmin = 3 )
# B = np.array([1, 0, 5] , ndmin = 3 )

# B = np.array([1, 0, 5] , ndmin = 3 ) 




# # Assuming 1 variable only
# def distance_point_DNF ( P)
    
#     for C in P:
#         for ap in C:
#             if (C[1] == 0)


# print(dnf)