from z3 import *
import numpy as np
from scipy.optimize import minimize, LinearConstraint

s = 10

K1 = 1
K2 = 2
K3 = 1

SB = 5
SD = 3

max_guesses = 1000

max_disjuncts = 2
max_conjuncts = 3

# Print warnings or not?
ON = 1
OFF = 0
DISPLAY_WARNINGS = ON

# Number of variables
n = 3

# Only for 3 variable case
x, y, z, xp, yp, zp = Ints('x y z xp yp zp')


class partial_transition_function:
    def __init__(self, DNF , transition_matrix):
        self.b = DNF
        self.t = transition_matrix

def total_transition_function (A):
    return [ partial_transition_function(np.array([0,0,0,0,0], ndmin = 3), A ) ]


# Code Input
P_array = np.array( [ [[1,0,0,0,1] , [0,1,0,0,1] ], [[0,0,1,0,1] , [0,0,0,0,0]] , [[0,0,1,0,0] , [0,0,0,0,0]] ])
B_array = np.array( [ [[0,1,0,-2,1000] , [0,0,1,0,1] ] ] )
Q_array = np.array( [ [[1,-1,0,1,0] ] ] )
# T_function = [partial_transition_function(np.array([1,1,1,2,0] , ndmin = 3), np.array( [[1,2,3,1], [2,3,1,4] , [1,3,1,4], [0,0,0,1]] , ndmin = 2 )), 
    # partial_transition_function(np.array([1,1,1,-1,0], ndmin = 3), np.array( [[2,2,3,2], [2,3,2,4] , [2,3,3,4], [0,0,0,1]] , ndmin = 2 ) )] 
T_function = total_transition_function(np.array( [[1,1,0,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]] , ndmin = 2 ))




''' ********************************************************************************************************************'''
# for 3D case only
def convert_predicate_to_lambda ( P  ):
    if (P[n] == 0):
        return lambda x,y,z : int(P[0]) * x + int(P[1]) * y + int(P[2]) * z == int(P[n+1])
    elif (P[n] == -1):
        return lambda x,y,z : int(P[0]) * x + int(P[1]) * y + int(P[2]) * z <= int(P[n+1])
    elif (P[n] == -2):
        return lambda x,y,z : int(P[0]) * x + int(P[1]) * y + int(P[2]) * z < int(P[n+1])
    elif (P[n] == 1):
        return lambda x,y,z : int(P[0]) * x + int(P[1]) * y + int(P[2]) * z >= int(P[n+1])
    elif (P[n] == 2):
        return lambda x,y,z : int(P[0]) * x + int(P[1]) * y + int(P[2]) * z > int(P[n+1])
    else:
        print("Incorrect value to predicate array operator")
        return lambda x,y,z : False

# Testing the function:
# S = convert_predicate_to_lambda(np.array([1,2,3,0,0], ndmin = 1) )
# print(S(-1,2,3), S(-2,1,3) , S(0,0,0) , S(1,1,1) , S(2,1,3))

def convert_conjunctiveClause_to_lambda (C):
    if np.size(C) == 0:
        return lambda x,y,z: True
    return lambda x,y,z: And(convert_predicate_to_lambda( C[0])(x,y,z), convert_conjunctiveClause_to_lambda( C[1:])(x,y,z) )

# Testing the function:
# S = convert_conjunctiveClause_to_lambda( np.array( [[1,2,3,0,0] , [1,1,-1,0,-1] ], ndmin = 2) )
# print(S(-1,2,3), S(-2,1,3) , S(0,0,0) , S(1,1,1) , S(2,1,3))

def convert_DNF_to_lambda (D):
    if np.size(D) == 0:
        return lambda x,y,z: False
    return lambda x,y,z: Or(convert_conjunctiveClause_to_lambda( D[0])(x,y,z), convert_DNF_to_lambda( D[1:])(x,y,z) )

# Testing the function:
# P1 = np.array( [ [[1,2,3,0,0] , [1,2,3,0,-1]] , [[1,3,3,2,1] , [0,0,0,0,0] ] ] ) 
# S3 = convert_DNF_to_lambda( P1 )
# print(S3(-2,1,3)) #To simplify expression, use simplify( .) function.


# A is a (n+1) * (n+1) matrix.
def convert_transition_matrix_to_lambda (A):
    return lambda x, y, z, xp, yp, zp: And(xp == int(A[0,0])*x + int(A[0,1])*y + int(A[0,2])*z + int(A[0,1]) , yp == int(A[1,0])*x + int(A[1,1])*y + int(A[1,2])*z + int(A[1,1])
                                                            , zp == int(A[2,0])*x + int(A[2,1])*y + int(A[2,2])*z + int(A[2,1])   )

def convert_transition_function_to_lambda (f):
    if len(f) == 0:
        return lambda x, y, z, xp, yp, zp: True
    else: 
        return lambda x, y, z, xp, yp, zp: xp == If( convert_DNF_to_lambda(f[0].b)(x,y,z), convert_transition_matrix_to_lambda(f[0].t)(x,y,z,xp,yp,zp), 
                                                                            convert_transition_function_to_lambda(f[1:])(x,y,z,xp,yp,zp) )




# Testing the functions.
# A = np.array( [[1,2,3,1], [2,3,1,4] , [1,3,1,4], [0,0,0,1]] , ndmin = 2 )
# X = convert_transition_matrix_to_lambda(A)
# print(X(2,1,2,3,2,1), X(3,2,1,3,3,1))
# S = total_transition_function(A)
# print(S[0].b, S[0].t)
''' ********************************************************************************************************************'''

''' ********************************************************************************************************************'''
# 'COC' is short for coefficients, operators, constants
def extract_COC_from_predicate(P):
    return (  np.concatenate((P[0:n], P[n+1:] )) , P[n:n+1]   )

def extract_COC_from_conjunctiveClause_internal(C):
    if np.size(C) == 0:
        return [ np.empty(0), np.empty(0) ]
    COC1 = extract_COC_from_predicate(C[0])
    COC2 = extract_COC_from_conjunctiveClause_internal(C[1:])
    return  [ np.concatenate(( COC1[0] , COC2[0] ))  , np.concatenate((COC1[1], COC2[1]))  ]

def extract_COC_from_conjunctiveClause(C):
    A = extract_COC_from_conjunctiveClause_internal(C)
    A[0] = (A[0].reshape(-1, n+1)).astype(int)
    A[1] = A[1].astype(int)
    return A

def get_predicate_from_COC ( cc , o):
    return np.concatenate((np.concatenate((cc[0:n] , np.array([o]) )) , cc[n:n+1] ))


def get_DNF_from_COC_internal ( CC, O):
    if np.size(O) == 0:
        return np.empty(0)
    A1 = get_predicate_from_COC( CC[0], O[0]  )
    A2 = get_DNF_from_COC_internal (CC[1: ] , O[1: ])
    return np.concatenate (( A1, A2 ))

def get_DNF_from_COC (COC):
    CC = COC[0]
    O = COC[1]
    A = get_DNF_from_COC_internal(CC, O)
    A = (A.reshape(-1, n+2)).astype(int)
    return A

# print(extract_COC_from_predicate(np.array([6,2,3,1,6]) ) )
# print(extract_COC_from_conjunctiveClause( np.array([[1,2,3,1,3], [4,2,3,-1,6] , [7,3,3,-2,9]] ) ) )
# print(get_predicate_from_COC( np.array([1,2,3,3]) , 2) )
# print(get_DNF_from_COC( [np.array( [[1,2,3,3], [4,2,3,6], [7,2,3,9]] ), np.array([2,1,-1]) ] ) )
# print (get_DNF_from_COC(extract_COC_from_conjunctiveClause( np.array([[1,1,1,-1,3], [4,5,3,-2,6] , [7,8,3,-1,9]] ))) )


# This assumes that T_function is a bijective function i.e. each partial transition matrix is non-singular, and that their respective domains are bijective (clearly too strong!)
# Also inverse matrix will not always have integer values!!!

def inverse_transition_function (f):
    if len(f) == 0:
        return []
    transition_matrix = f[0].t
    if (np.linalg.det(f[0].t) ):
        inverse_partial_transition_matrix = np.linalg.inv(f[0].t).astype(int) #Non-Singular Matrix
    else:
        if (DISPLAY_WARNINGS):
            print("Transition function has a singular partial transition matrix, inverse isn't defined")
        inverse_partial_transition_matrix = np.eye(n+1) #Singular Matrix, inverse doesn't exist!
    DNF = f[0].b

    new_DNF = np.empty(0)
    for C in DNF:
        temp = extract_COC_from_conjunctiveClause(C)
        temp[0] = np.dot(temp[0], inverse_partial_transition_matrix.transpose() )
        new_CC = get_DNF_from_COC( temp )
        if (np.size(new_DNF) == 0):
            new_DNF = new_CC.reshape(1,-1,n+2)
        else:
            new_DNF = np.append(new_DNF, new_CC.reshape(1,-1,n+2), axis = 0)
    return [partial_transition_function( new_DNF , inverse_partial_transition_matrix )] + inverse_transition_function(f[1:]) 


''' ********************************************************************************************************************'''

P = convert_DNF_to_lambda(P_array)
B = convert_DNF_to_lambda(B_array)
Q = convert_DNF_to_lambda(Q_array)
T_inv = inverse_transition_function(T_function)
T = convert_transition_function_to_lambda( T_function)

''' ********************************************************************************************************************'''
def get_all_constants_in_2D_array( A ):
    temp = A.tolist()
    rv = []
    for a in temp:
        rv = rv + a 
    return list(set(rv))

def get_all_constants_in_3D_array( A ):
    if len(A) == 0:
        return []
    return list(set(get_all_constants_in_2D_array(A[0]) + get_all_constants_in_3D_array(A[1:])))

def get_all_program_constants():
    rv = list(set(get_all_constants_in_3D_array(P_array) + get_all_constants_in_3D_array(B_array) + get_all_constants_in_3D_array(Q_array)))
    for partial in T_function:
        rv = list(set(rv + get_all_constants_in_3D_array(partial.b) + get_all_constants_in_2D_array(partial.t)))
    return rv

programConstants = get_all_program_constants()
''' ********************************************************************************************************************'''

''' ********************************************************************************************************************'''
# Let normalized versions only have <= or < operators.
def normalize_predicate (P):
    if (P[n] > 0):
        return np.array( np.multiply(P, -1), ndmin = 2)
    elif (P[n] == 0):
        temp1, temp2 = np.multiply(P, -1) , P
        temp1[n] = -1
        temp2[n] = -1
        return np.array([temp1, temp2], ndmin = 2 )
    return np.array(P, ndmin = 2)

def normalize_conjunctiveClause (C):
    if (len(C) == 0):
        return np.empty(shape = (0,n+2), dtype = int)
    return np.concatenate(( normalize_predicate(C[0]) , normalize_conjunctiveClause(C[1:]) ))

def normalize_DNF_internal (D, conjunct_size): 
    if (len(D) == 0):
        return np.empty(shape = (0,conjunct_size,n+2), dtype = int)
    C = normalize_conjunctiveClause(D[0])
    padding_predicate = np.array([0,0,0,-1,0], ndmin = 1)
    while (len(C) < conjunct_size):
        C = np.concatenate((C, np.array( padding_predicate, ndmin = 2) ))
    return np.concatenate(( np.array(C, ndmin = 3) , normalize_DNF_internal(D[1:], conjunct_size) ))

def normalize_DNF (D):
    max_size = 0
    for C in D:
        curr_size = 0
        for P in C:
            curr_size = curr_size + ( 2 if (P[n] == 0) else 1)
        max_size = max(max_size, curr_size)
    return normalize_DNF_internal(D, max_size)


# print(normalize_predicate(np.array([1,2,0,-1,4])))
# print(normalize_conjunctiveClause( np.array( [ [1,2,3,1,10], [1,3,1,0,0] , [1,3,1,-2,0] ] , ndmin = 2)) ) 
# print(normalize_DNF(np.array( [ [[1,0,0,0,1] , [0,1,0,0,1] ], [[0,0,1,0,1] , [0,0,0,-2,0]] , [[0,0,1,-1,0] , [0,0,0,1,0]] ]) )  )
''' ********************************************************************************************************************'''


''' ********************************************************************************************************************'''

def guess_invariant_smallConstants (maxConstantValue, max_conjunctive_clause_size, max_number_of_disjuncts, operator_probability_matrix):
    conjunctive_clause_size = np.random.randint(1, max_conjunctive_clause_size + 1 )
    number_of_disjuncts = np.random.randint(1, max_number_of_disjuncts + 1)
    I = np.random.randint(-1 * maxConstantValue, maxConstantValue + 1, size=(number_of_disjuncts, conjunctive_clause_size, n+2))
    for i in range(number_of_disjuncts):
        for j in range(conjunctive_clause_size):
            I[i, j, n] = np.random.choice([-2, -1, 0, 1, 2], p=operator_probability_matrix)
            while ( np.array_equal( I[i, j, 0:n] , np.zeros(n) ) ):  # Ensure that all coefficients of a predicate are not zero.
                newcoefficients = p.random.randint(-1 * maxConstantValue, maxConstantValue + 1, size=(n) )
                I[i, j, 0] = newcoefficients[0]
                I[i, j, 1] = newcoefficients[1]
                I[i, j, 2] = newcoefficients[2]
    return I

def guess_invariant_octagonaldomain ( listOfConstants, max_conjunctive_clause_size, max_number_of_disjuncts, operator_probability_matrix):
    conjunctive_clause_size = np.random.randint(1, max_conjunctive_clause_size + 1 )
    number_of_disjuncts = np.random.randint(1, max_number_of_disjuncts + 1)
    I = np.random.choice(listOfConstants, size=(number_of_disjuncts, conjunctive_clause_size, n + 2))
    for i in range(number_of_disjuncts):
        for j in range(conjunctive_clause_size):
            I[i, j, n] = np.random.choice([-2, -1, 0, 1, 2], p=operator_probability_matrix)
            nonzero_coefficient_position = np.random.randint(0, n, size = (2))
            nonzero_coefficient_value = np.random.choice([-1,1], p = [0.5, 0.5], size = (2))
            for k in range(n):
                if k in nonzero_coefficient_position:
                    position = np.where(nonzero_coefficient_position == k) #position is a pair of a (list representing all positions in nonzero_coefficient_position which have value k) and the type of the list.
                    I[i,j,k] = nonzero_coefficient_value[position[0][0]]
                else:
                    I[i,j,k] = 0
    return I

def guess_invariant_octagonaldomain_extended ( listOfConstants, max_conjunctive_clause_size, max_number_of_disjuncts, operator_probability_matrix):
    conjunctive_clause_size = np.random.randint(1, max_conjunctive_clause_size + 1 )
    number_of_disjuncts = np.random.randint(1, max_number_of_disjuncts + 1)
    I = np.random.choice(listOfConstants, size=(number_of_disjuncts, conjunctive_clause_size, n + 2))
    for i in range(number_of_disjuncts):
        for j in range(conjunctive_clause_size):
            I[i, j, n] = np.random.choice([-2, -1, 0, 1, 2], p=operator_probability_matrix)
            newcoefficients = np.zeros(n)
            while (np.array_equal( newcoefficients , np.zeros(n)) ): # Ensure that all coefficients of a predicate are not zero.
                newcoefficients = np.random.choice([-1,0,1], p=[0.33,0.34, 0.33], size = (n))
                I[i, j, 0] = newcoefficients[0]
                I[i, j, 1] = newcoefficients[1]
                I[i, j, 2] = newcoefficients[2]
    return I

def guess_invariant_nearProgramConstants ( constants_in_program, k,  max_conjunctive_clause_size, max_number_of_disjuncts, operator_probability_matrix):
    conjunctive_clause_size = np.random.randint(1, max_conjunctive_clause_size + 1 )
    number_of_disjuncts = np.random.randint(1, max_number_of_disjuncts + 1)
    list_of_Constants = []
    for constant in constants_in_program:
        list_of_Constants = list_of_Constants + list(range(constant - k, constant + k + 1))
    list_of_Constants = list(set(list_of_Constants)) #remove duplicates
    I = np.random.choice(list_of_Constants, size=(number_of_disjuncts, conjunctive_clause_size, n+2))
    for i in range(number_of_disjuncts):
        for j in range(conjunctive_clause_size):
            I[i, j, n] = np.random.choice([-2, -1, 0, 1, 2], p=operator_probability_matrix)
            while ( np.array_equal( I[i, j, 0:n] , np.zeros(n) ) ):  # Ensure that all coefficients of a predicate are not zero.
                newcoefficients = np.random.choice(list_of_Constants, size=(n) )
                I[i, j, 0] = newcoefficients[0]
                I[i, j, 1] = newcoefficients[1]
                I[i, j, 2] = newcoefficients[2]
    return I


# print(guess_invariant_smallConstants(10, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) ))
# print(guess_invariant_octagonaldomain(programConstants, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) ))
# print(guess_invariant_octagonaldomain_extended(programConstants, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) ))
# print(guess_invariant_nearProgramConstants(programConstants, 2, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) ))
''' ********************************************************************************************************************'''

''' ********************************************************************************************************************'''
def print_predicate (P):
    if (P[n] == 0):
        print("(%d*x + %d*y + %d*z == %d)" %(int(P[0]), int(P[1]), int(P[n-1]), int(P[n+1])) , end = '')
    elif (P[n] == -1):
        print("(%d*x + %d*y + %d*z <= %d)" %(int(P[0]), int(P[1]), int(P[n-1]), int(P[n+1])) , end = '')
    elif (P[n] == -2):
        print("(%d*x + %d*y + %d*z < %d)" %(int(P[0]), int(P[1]), int(P[n-1]), int(P[n+1])) , end = '')
    elif (P[n] == 1):
        print("(%d*x + %d*y + %d*z >= %d)" %(int(P[0]), int(P[1]), int(P[n-1]), int(P[n+1])) , end = '')
    elif (P[n] == 2):
        print("(%d*x + %d*y + %d*z > %d)" %(int(P[0]), int(P[1]), int(P[n-1]), int(P[n+1])) , end = '')
    else:
        print("Incorrect value to predicate array operator")
    return

def print_conjunctiveClause (C):
    if (len(C) == 0):
        return
    elif (len(C) == 1):
        print_predicate(C[0])
    else:
        print("And(", end = '')
        for i in range(len(C)):
            print_predicate(C[i])
            if (i != len(C) - 1):
                print(",", end = '')
        print(")", end = '')
    return

# s = 0 indicates no new line after printing DNF.
def print_DNF (D, s):
    if (len(D) == 0):
        return
    elif (len(D) == 1):
        print_conjunctiveClause(D[0])
    else:    
        print("Or( ", end = '')
        for i in range(len(D)):
            print_conjunctiveClause(D[i])
            if (i != len(D) - 1):
                print(", ", end = '')
        if (s == 0):
            print(" )", end = '')
        else:
            print(" )")
    return    

# Testing
# print_predicate(np.array([1,2,3,0,0], ndmin = 1))
# print_conjunctiveClause(np.array( [[1,2,3,0,0], [1,3,4,1,2] , [1,1,3,-2,3]] , ndmin = 2))
# print_DNF(np.array( [[[1,2,3,0,0], [1,1,3,1,2] , [1,2,3,-2,3]] , [[1,1,2,0,43], [1,1,1,-1,2] , [1,1,2,2,3]]] , ndmin = 3), 0)
''' ********************************************************************************************************************'''

''' ********************************************************************************************************************'''
def C1(I):
    return Implies(P(x,y,z), I(x,y,z))

def C2(I):
    return Implies(And(B(x,y,z), I(x,y,z), T(x,y,z, xp,yp,zp)) , I(xp, yp, zp))

def C3(I):
    return Implies(And(I(x,y,z), Not(B(x,y,z))), Q(x,y,z))

def System(I):
    return And(C1(I), C2(I), C3(I))

# Returns true or a counterexample
def Check(C, I):
    s = Solver()
    # Add the negation of the conjunction of constraints
    s.add(Not(C))
    r = s.check()
    output = r.__repr__()
    if output == "sat":
        return s.model()
    elif output == "unsat":
        return None
    else:
        print("Solver can't verify or disprove, it says: %s for invariant %s" %(r, I))
        return None

def GenerateCexList_C1 (I , number_of_cex):
    if (number_of_cex == 0):
        return []
    cex = Check(C1(I), I)
    if cex is None:
        return []
    else:
        return [cex] + GenerateCexList_C1( lambda u,v,w: Or( I(u,v,w), And(u == cex.evaluate(x, model_completion=True), 
                                                            v == cex.evaluate(y, model_completion=True), w == cex.evaluate(z, model_completion=True) ) ) ,  number_of_cex - 1  )

# There is another way to generate cex for this case which is not been coded and that is by excluding both x and xp points.
def GenerateCexList_C2 (I , number_of_cex):
    if (number_of_cex == 0):
        return []
    cex = Check(C2(I), I)
    if cex is None:
        return []
    else:
        return [cex] + GenerateCexList_C2( lambda u,v,w: Or( I(u,v,w), And(u == cex.evaluate(x, model_completion=True), v == cex.evaluate(y, model_completion=True), 
                                                                        w == cex.evaluate(z, model_completion=True) ), 
                                                            And(u == cex.evaluate(xp, model_completion=True), v == cex.evaluate(yp, model_completion=True), 
                                                                        w == cex.evaluate(zp, model_completion=True) )) ,  number_of_cex - 1  )

def GenerateCexList_C3 (I , number_of_cex):
    if (number_of_cex == 0):
        return []
    cex = Check(C3(I), I)
    if cex is None:
        return []
    else:
        return [cex] + GenerateCexList_C3( lambda u,v,w: And( I(u,v,w), Not(And(u == cex.evaluate(x, model_completion=True), 
                                                            v == cex.evaluate(y, model_completion=True), w == cex.evaluate(z, model_completion=True) ) ) )  ,  number_of_cex - 1   )


# I_g_array = guess_invariant_smallConstants(10, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) )
# I_g = convert_DNF_to_lambda(I_g_array)
# print(GenerateCexList_C1(I_g, 10) ,'\n')
# print(GenerateCexList_C2(I_g, 10), '\n')
# print(GenerateCexList_C3(I_g, 10), '\n')


''' ********************************************************************************************************************'''

''' ********************************************************************************************************************'''
'''Operator code:
g = 2
ge = 1
eq = 0
le = -1
l = -2 '''

def dist(x, p):
    return np.linalg.norm(x - p)

def distance_point_conjunctiveClause(p, C):
    C = normalize_conjunctiveClause(C)
    A = extract_COC_from_conjunctiveClause(C)[0]
    return float(minimize(
        dist, 
        np.zeros(3),
        args=(p,),
        constraints=[LinearConstraint(A[:, :-1], -np.inf, -A[:, -1])],
    ).fun)


def distance_point_DNF(p, D):
    d = float('inf')
    for C in D:
        d = min(d, distance_point_conjunctiveClause(p, C))
    return d

#Testing:
# print(distance_point_conjunctiveClause( np.array( [-3,1,3], ndmin = 1), np.array( [ [1,2,3,1,10], [1,3,1,0,0] ] , ndmin = 2)) ) 
# print(distance_point_DNF( np.array( [-3,3,1], ndmin = 1), np.array( [ [ [-7,1,3,-2,3], [1,2,1,0,2], [3,1,3, 1, 4] ], [ [1,3,1,1,10], [1,3,1,0,0], [0,0,0,0,0] ] ]    )) )


''' ********************************************************************************************************************'''

''' ********************************************************************************************************************'''
# This is specific for this clause system.

def sample_points_from_DNF ( D , number_of_points, sample_list):
    if (number_of_points == 0):
        return sample_list

    s = Solver()
    s.add(Implies(True, D(x,y,z)))
    r = s.check()
    output = r.__repr__()
    if output == "sat":
        sample_point = s.model()
        sample_list.append(sample_point) 
        return sample_points_from_DNF( lambda u,v,w: And( D(u,v,w), Or(u != sample_point.evaluate(x), v != sample_point.evaluate(y), w != sample_point.evaluate(z) ) ) 
                                                    , number_of_points - 1, sample_list)
    elif output == "unsat":
        return sample_list
    else:
        print("Sampler can't sample, it says: %s" %(r))
        return sample_list

# Assumes transition function is a total function.
def unroll_chain_from_starting_point ( pt_matrix, transition_function, conditional_predicate, number_of_points, sample_list):
    if (number_of_points == 0):
        return sample_list
    if (simplify(conditional_predicate(int(pt_matrix[0]), int(pt_matrix[1]), int(pt_matrix[2]) ) )):
        sample_list.append(pt_matrix[0:n]) 
        new_pt_matrix = np.empty( n + 1 , int)
        for partial_tf in transition_function:
            if ( simplify(convert_DNF_to_lambda(partial_tf.b)(int(pt_matrix[0]), int(pt_matrix[1]), int(pt_matrix[2]) )) ):
                new_pt_matrix[:] = np.dot(  pt_matrix, np.transpose(partial_tf.t) )
        return unroll_chain_from_starting_point(  new_pt_matrix, transition_function, conditional_predicate, number_of_points - 1 , sample_list)
    else:
        return sample_list

def get_positive_points( sampling_breadth, sampling_depth):
    temp_list = []
    sample_points_from_DNF(P, sampling_breadth, temp_list)
    breadth_list_of_positive_points = []
    for sample in temp_list:
        pt_x = sample.evaluate(x).as_long()
        pt_y = sample.evaluate(y).as_long()
        pt_z = sample.evaluate(z).as_long()
        breadth_list_of_positive_points.append(np.array([pt_x, pt_y, pt_z]))
    list_of_positive_points = []
    for pt in breadth_list_of_positive_points:
        pt_matrix = np.concatenate((pt, np.array([1])))
        unroll_chain_from_starting_point(pt_matrix, T_function, B, SD + 1, list_of_positive_points)

    uniques = [] #remove duplicates
    for arr in list_of_positive_points:
        if not any(np.array_equal(arr, unique_arr) for unique_arr in uniques):
            uniques.append(arr)
    return uniques


def get_negative_points( sampling_breadth, sampling_depth):
    temp_list = []
    sample_points_from_DNF( lambda x,y,z: And( Not(Q(x,y,z)), Not(B(x,y,z)) ) , sampling_breadth, temp_list)
    breadth_list_of_negative_points = []
    for sample in temp_list:
        pt_x = sample.evaluate(x).as_long()
        pt_y = sample.evaluate(y).as_long()
        pt_z = sample.evaluate(z).as_long()
        breadth_list_of_negative_points.append(np.array([pt_x, pt_y, pt_z]))
    list_of_negative_points = []
    for pt in breadth_list_of_negative_points:
        pt_matrix = np.concatenate((pt, np.array([1])))
        unroll_chain_from_starting_point(pt_matrix, T_inv , lambda x,y,z: Or( Not(Q(x,y,z)), B(x,y,z) ), SD + 1, list_of_negative_points )
    
    uniques = [] #remove duplicates
    for arr in list_of_negative_points:
        if not any(np.array_equal(arr, unique_arr) for unique_arr in uniques):
            uniques.append(arr)
    return uniques

# Testing these functions
# print(get_positive_points(SB, SD))
# print(get_negative_points(SB, SD))

''' ********************************************************************************************************************'''

''' ********************************************************************************************************************'''

def J1(I, cexList):
    num = len(cexList)
    error = 0
    for cex in cexList: 
        pt_x = cex.evaluate(x, model_completion=True).as_long()
        pt_y = cex.evaluate(y, model_completion=True).as_long()
        pt_z = cex.evaluate(z, model_completion=True).as_long() 
        point = np.array( [pt_x, pt_y, pt_z ], ndmin = 1) 
        error = max(error, distance_point_DNF(point, I))
    return error + num

# Traditionally try to 'guess' which cex are supposed to be negative, and which are supposed to be positive, and then there is a relative ratio; but we skip that here.
def J2(cexList):
    num = len(cexList)
    return num

def J3(Q, cexList):
    num = len(cexList)
    error = 0
    for cex in cexList:
        pt_x = cex.evaluate(x).as_long()
        pt_y = cex.evaluate(y).as_long()
        pt_z = cex.evaluate(z).as_long()
        point = np.array( [pt_x, pt_y, pt_z ], ndmin = 1) 
        error = max(error, distance_point_DNF(point, Q))
    return error + num

# Testing these functions
# I_g_array = guess_invariant_smallConstants(10, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) )
# I_g = convert_DNF_to_lambda(I_g_array)
# cexList1 = GenerateCexList_C1(I_g, 10)
# cexList2 = GenerateCexList_C2(I_g, 10)
# cexList3 = GenerateCexList_C3(I_g, 10)
# print(J1(I_g_array, cexList1))
# print(J2(cexList2))
# print(J3(Q_array, cexList3))

''' ********************************************************************************************************************'''

# Here in all our guess strategies, we assume that conjunctive clause size is same.

'''
guess_strategy code:
(1,k) -> smallConstants(k)
(2,0) -> octagonaldomain
(2,1) -> octagonaldomain_extended
(3,k) -> nearProgramConstants(k)
'''

def random_invariant_guess (guesses, guess_strategy, no_of_conjuncts , no_of_disjuncts):
    cost = float('inf')
    count = 0
    while (cost != 0 and count < guesses):
        count = count + 1
        if (guess_strategy[0] == 1):
            I_g_array = guess_invariant_smallConstants(guess_strategy[1], no_of_conjuncts, no_of_disjuncts, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) )
        elif (guess_strategy[0] == 2):
            if (guess_strategy[1] == 0):
                I_g_array = guess_invariant_octagonaldomain(programConstants, no_of_conjuncts, no_of_disjuncts, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
            else:
                I_g_array = guess_invariant_octagonaldomain_extended(programConstants, no_of_conjuncts, no_of_disjuncts, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
        elif (guess_strategy[0] == 3):
            I_g_array = guess_invariant_nearProgramConstants(programConstants, guess_strategy[1], no_of_conjuncts, no_of_disjuncts,  np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
        I_g = convert_DNF_to_lambda(I_g_array)
        print(count,'   ', end = '')
        print_DNF(I_g_array, 0)
        print("\t", end = '')

        C1_cexList = GenerateCexList_C1 (I_g, s)
        # print(C1_cexList)
        C2_cexList = GenerateCexList_C2 ( I_g, s)
        # print(C2_cexList)
        C3_cexList = GenerateCexList_C3 ( I_g, s)
        # print(C3_cexList)

        # Get costFunction
        cost1 = J1(I_g_array, C1_cexList)
        cost2 = J2(C2_cexList)
        cost3 = J3(Q_array, C3_cexList)
        cost = K1*cost1 + K2*cost2 + K3*cost3

        # print(cost1, cost2, cost3)
        print('   ', cost,'\n')
    return


random_invariant_guess(max_guesses, (3,1), max_conjuncts, max_disjuncts )
