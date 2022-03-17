from z3 import *
import numpy as np

s = 10

K1 = 1
K2 = 2
K3 = 1

SB = 2
SD = 5

max_time = 5000

# Only for 1 variable case
x, xp = Ints('x xp')


''' ********************************************************************************************************************'''
# for 1D case only
def convert_predicate_to_lambda ( P  ):
    if (P[1] == 0):
        return lambda x : int(P[0]) * x == int(P[2])
    elif (P[1] == -1):
        return lambda x : int(P[0]) * x <= int(P[2])
    elif (P[1] == -2):
        return lambda x : int(P[0]) * x < int(P[2])
    elif (P[1] == 1):
        return lambda x : int(P[0]) * x >= int(P[2])
    elif (P[1] == 2):
        return lambda x : int(P[0]) * x > int(P[2])
    else:
        print("Incorrect value to predicate array operator")
        return lambda x: False

# Testing the function:
# S = convert_predicate_to_lambda(np.array([1,0,0], ndmin = 1) )
# print(S(-1), S(-2) , S(0) , S(1) , S(2))

def convert_conjunctiveClause_to_lambda (C):
    if np.size(C) == 0:
        return lambda x: True
    return lambda t: And(convert_predicate_to_lambda( C[0])(t), convert_conjunctiveClause_to_lambda( C[1:])(t) )

# Testing the function:
# S2 = convert_conjunctiveClause_to_lambda( np.array( [[1,0,0] , [1,0,-1] ], ndmin = 2) )
# print(S2(-2), S2(-1) , S2(0) , S2(1) , S2(2))

def convert_DNF_to_lambda (D):
    if np.size(D) == 0:
        return lambda x: False
    return lambda t: Or(convert_conjunctiveClause_to_lambda( D[0])(t), convert_DNF_to_lambda( D[1:])(t) )

def convert_transition_matrix_to_lambda (A):
    return lambda x, xp: xp == int(A[0,0])*x + int(A[0,1])

# Testing the function:
# P1 = np.array( [ [[1,0,0] , [1,0,-1]] , [[1,2,1]] ] , dtype=object) 
# S3 = convert_DNF_to_lambda( P1 )
# print(S3(-2)) #To simplify expression, use simplify( .) function.
# A = np.array( [[1,1], [0,1]] , ndmin = 2 )
# X = convert_transition_matrix_to_lambda(A)
# print(X(2,3), X(3,3))
''' ********************************************************************************************************************'''

# Although same code as *_lambda.py, this always works (never repeats cex); it starting repeating cex after I switched from lambda to converted lambda form.

programConstants = [-2,0,1,5]

P_array = np.array( [ [[1,0,0] ] ] , dtype=object)
B_array = np.array( [ [[1,-2,5] ] ] , dtype=object)
Q_array = np.array( [ [[1,0,5] ] ] , dtype=object)
T_array = np.array( [[1,1], [0,1]] , ndmin = 2 )

P = convert_DNF_to_lambda(P_array)
B = convert_DNF_to_lambda(B_array)
Q = convert_DNF_to_lambda(Q_array)
T = convert_transition_matrix_to_lambda(T_array)

''' ********************************************************************************************************************'''


def guess_invariant_smallConstants (maxConstantValue, max_conjunctive_clause_size, max_number_of_disjuncts, operator_probability_matrix):
    conjunctive_clause_size = np.random.randint(1, max_conjunctive_clause_size + 1 )
    number_of_disjuncts = np.random.randint(1, max_number_of_disjuncts + 1)
    I = np.random.randint(-1 * maxConstantValue, maxConstantValue + 1, size=(number_of_disjuncts, conjunctive_clause_size, 3))
    for i in range(number_of_disjuncts):
        for j in range(conjunctive_clause_size):
            I[i, j, 1] = np.random.choice([-2, -1, 0, 1, 2], p=operator_probability_matrix)
            if (I[i, j, 0] == 0):  # Ensure that all coefficients of a predicate are not zero.
                I[i, j, 0] = np.random.choice([i for i in range(-1 * maxConstantValue, maxConstantValue + 1) if i not in [0]])
    return I

def guess_invariant_octagonaldomain ( listOfConstants, max_conjunctive_clause_size, max_number_of_disjuncts, operator_probability_matrix):
    conjunctive_clause_size = np.random.randint(1, max_conjunctive_clause_size + 1 )
    number_of_disjuncts = np.random.randint(1, max_number_of_disjuncts + 1)
    I = np.random.choice(listOfConstants, size=(number_of_disjuncts, conjunctive_clause_size, 3))
    for i in range(number_of_disjuncts):
        for j in range(conjunctive_clause_size):
            I[i, j, 1] = np.random.choice([-2, -1, 0, 1, 2], p=operator_probability_matrix)
            I[i, j, 0] = np.random.choice([-1,1], p=[0.5,0.5]) #Typically octagonal domain includes constants +1, 0 and -1, but all coefficients cannot be 0; and here there is only 1 variable
    return I

def guess_invariant_nearProgramConstants ( constants_in_program, k,  max_conjunctive_clause_size, max_number_of_disjuncts, operator_probability_matrix):
    conjunctive_clause_size = np.random.randint(1, max_conjunctive_clause_size + 1 )
    number_of_disjuncts = np.random.randint(1, max_number_of_disjuncts + 1)
    list_of_Constants = []
    for constant in constants_in_program:
        list_of_Constants = list_of_Constants + list(range(constant - k, constant + k + 1))
    list_of_Constants = list(set(list_of_Constants)) #remove duplicates
    I = np.random.choice(list_of_Constants, size=(number_of_disjuncts, conjunctive_clause_size, 3))
    for i in range(number_of_disjuncts):
        for j in range(conjunctive_clause_size):
            I[i, j, 1] = np.random.choice([-2, -1, 0, 1, 2], p=operator_probability_matrix)
            if (I[i, j, 0] == 0):  # Ensure that all coefficients of a predicate are not zero.
                I[i, j, 0] = np.random.choice([i for i in list_of_Constants if i not in [0]])
    return I


# print(guess_invariant_smallConstants(10, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) ))
''' ********************************************************************************************************************'''

''' ********************************************************************************************************************'''
def print_predicate (P):
    if (P[1] == 0):
        print("(%d*x == %d)" %(int(P[0]), int(P[2])) , end = '')
    elif (P[1] == -1):
        print("(%d*x <= %d)" %(int(P[0]), int(P[2])) , end = '')
    elif (P[1] == -2):
        print("(%d*x < %d)" %(int(P[0]), int(P[2])) , end = '')
    elif (P[1] == 1):
        print("(%d*x >= %d)" %(int(P[0]), int(P[2])) , end = '')
    elif (P[1] == 2):
        print("(%d*x > %d)" %(int(P[0]), int(P[2])) , end = '')
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
# print_predicate(np.array([1,0,0], ndmin = 1))
# print_conjunctiveClause(np.array( [[1,0,0], [1,1,2] , [1,-2,3]] , ndmin = 2))
# print_DNF(np.array( [[[1,0,0], [1,1,2] , [1,-2,3]] , [[1,0,43], [1,-1,2] , [1,2,3]]] , ndmin = 3))
''' ********************************************************************************************************************'''


def C1(I):
    return Implies(P(x), I(x))

def C2(I):
    return Implies(And(B(x), I(x), T(x, xp)) , I(xp))

def C3(I):
    return Implies(And(I(x), Not(B(x))), Q(x))

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
        return [cex] + GenerateCexList_C1( lambda t: Or( I(t), t == cex.evaluate(x) ) ,  number_of_cex - 1  )

# There is another way to generate cex for this case which is not been coded and that is by excluding both x and xp points.
def GenerateCexList_C2 (I , number_of_cex):
    if (number_of_cex == 0):
        return []
    cex = Check(C2(I), I)
    if cex is None:
        return []
    else:
        return [cex] + GenerateCexList_C2( lambda t: Or( I(t), t == cex.evaluate(x), t == cex.evaluate(xp) ) ,  number_of_cex - 1  )

def GenerateCexList_C3 (I , number_of_cex):
    if (number_of_cex == 0):
        return []
    cex = Check(C3(I), I)
    if cex is None:
        return []
    else:
        return [cex] + GenerateCexList_C3( lambda t: And( I(t), t != cex.evaluate(x) ) ,  number_of_cex - 1  )


''' ********************************************************************************************************************'''
'''Operator code:
g = 2
ge = 1
eq = 0
le = -1
l = -2 '''

P1 = np.array([1,0,0], ndmin = 3)

def distance_point_hyperplane ( p, L):
    x = float(p[0])
    L_endpoint_1 = float(L[2]) / float(L[0]) # For 1D only; for > 1 D, compute both endpoints, and distance is minimum of distance from these endpoints and also check distance of line and whether this distance lies within segment
    d = x - L_endpoint_1
    if (L[1] == 0):
        return abs(d)
    if ((L[1] > 0 and L[0]*d > 0) or (L[1] < 0 and L[0]*d < 0)):  # L[0]*d > 0 is short for (L[0]>0 and d > 0 ) or (L[0] < 0 and d < 0)
        d = 0
    return abs(d)

def distance_point_conjunctiveClause (p , C):
    d = float('inf')
    for L in C:
        d = min(d, distance_point_hyperplane(p, L))
    return d

def distance_point_DNF(p, D):
    d = float('inf')
    for C in D:
        d = min(d, distance_point_conjunctiveClause(p, C))
    return d

#Testing:
# print(distance_point_hyperplane( np.array( [1], ndmin = 1), np.array([2,-2,3] , ndmin = 1)) )
# print(distance_point_conjunctiveClause( np.array( [-3], ndmin = 1), np.array( [ [1,1,10], [1,0,0] ] , ndmin = 2)) ) 
# print(distance_point_DNF( np.array( [-3], ndmin = 1), np.array( [ [ [-7,-2,3], [1,0,2], [3, 1, 4] ], [ [1,1,10], [1,0,0] ] ] , dtype=object   )) ) #<-  d(-3, [(-7x < 3) /\ (x == 2) /\ (3x >= 4)] \/ [(x >= 10) /\ (x == 0)] )

''' ********************************************************************************************************************'''


''' ********************************************************************************************************************'''
# This is specific for this clause system.

def sample_points_from_DNF ( D , number_of_points, sample_list):
    if (number_of_points == 0):
        return sample_list

    s = Solver()
    s.add(Implies(True, D(x)))
    r = s.check()
    output = r.__repr__()
    if output == "sat":
        sample_point = s.model()
        sample_list.append(sample_point) 
        return sample_points_from_DNF( lambda t: And(D(t), t != sample_point.evaluate(x) ) , number_of_points - 1, sample_list)
    elif output == "unsat":
        return sample_list
    else:
        print("Sampler can't sample, it says: %s" %(r))
        return sample_list

def unroll_chain_from_starting_point ( pt_matrix, transition_matrix, conditional_predicate, number_of_points, sample_list):
    if (number_of_points == 0):
        return sample_list
    pt = simplify(pt_matrix[0])
    if (simplify(conditional_predicate(pt))):
        sample_list.append(pt.as_long())
        return unroll_chain_from_starting_point(  np.dot(transition_matrix, pt_matrix), transition_matrix, conditional_predicate, number_of_points - 1 , sample_list)
    else:
        return sample_list

# #Testing 'unroll_chain_from_starting_point' function
# p_m = np.array([0,1], dtype = object)
# final_list = []
# unroll_chain_from_starting_point(p_m, T_array, B, SD + 1, final_list)
# print(final_list)

def get_positive_points( sampling_breadth, sampling_depth):
    temp_list = []
    sample_points_from_DNF(P, sampling_breadth, temp_list)
    breadth_list_of_positive_points = []
    for sample in temp_list:
        breadth_list_of_positive_points.append(sample.evaluate(x))
    list_of_positive_points = []
    for pt in breadth_list_of_positive_points:
        pt_matrix = np.array([pt, 1], ndmin = 1, dtype = object)
        unroll_chain_from_starting_point(pt_matrix, T_array, B, SD + 1, list_of_positive_points)
    list_of_positive_points = list(set(list_of_positive_points)) #remove duplicates
    return list_of_positive_points


# At this moment, we assume that T_array^-1 is well defined i.e. T_array is non-singular
def get_negative_points( sampling_breadth, sampling_depth):
    temp_list = []
    sample_points_from_DNF( lambda t: And( Not(Q(t)), Not(B(t)) ) , sampling_breadth, temp_list)
    breadth_list_of_negative_points = []
    for sample in temp_list:
        breadth_list_of_negative_points.append(sample.evaluate(x))
    list_of_negative_points = []
    for pt in breadth_list_of_negative_points:
        pt_matrix = np.array([pt, 1], ndmin = 1, dtype = object)
        unroll_chain_from_starting_point(pt_matrix, np.linalg.inv(T_array).astype(int), lambda t: Or( Not(Q(t)), B(t) ), SD + 1, list_of_negative_points )
    list_of_negative_points = list(set(list_of_negative_points)) #remove duplicates
    return list_of_negative_points

# Testing these functions
# print(get_positive_points(SB, SD))
# print(get_negative_points(SB, SD))

''' ********************************************************************************************************************'''


''' ********************************************************************************************************************'''

def J1(I, cexList):
    error = 0
    for cex in cexList:
        point = np.array( [cex.evaluate(x).as_long() ], ndmin = 1, dtype = object) 
        error = max(error, distance_point_DNF(point, I))
    return error

# Traditionally try to 'guess' which cex are supposed to be negative, and which are supposed to be positive, and then there is a relative ratio; but we skip that here.
def J2(cexList):
    num = len(cexList)
    return num

def J3(Q, cexList):
    error = 0
    for cex in cexList:
        point = np.array([cex.evaluate(x).as_long() ], ndmin = 1, dtype = object) 
        error = max(error, distance_point_DNF(point, Q))
    return error


''' ********************************************************************************************************************'''

# Correct invariant is x <= 5
# Eg: I_g_array = np.array([1,0,0], ndmin = 3)

'''
mutation_strategy code:
1 -> smallConstants
2 -> octagonaldomain
3 -> nearProgramConstants
'''
# Implement timeout functionality!!!
def random_invariant_guess (timeout, mutation_strategy):
    cost = float('inf')
    while (cost != 0):
        if (mutation_strategy == 1):
            I_g_array = guess_invariant_smallConstants(6, 1, 1, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) )
        elif (mutation_strategy == 2):
            I_g_array = guess_invariant_octagonaldomain(programConstants, 1, 1, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
        elif (mutation_strategy == 3):
            I_g_array = guess_invariant_nearProgramConstants(programConstants, 1, 1, 1,  np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
        I_g = convert_DNF_to_lambda(I_g_array)

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
        print(cost)
    return


random_invariant_guess(3, 3)






