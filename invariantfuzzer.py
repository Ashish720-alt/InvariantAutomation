""" Imports.
"""
from z3 import *
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import operator


""" Hyper Parameters.
"""

s = 10

K1 = 1
K2 = 2
K3 = 1

SB = 5
SD = 3

max_guesses = 1000

max_disjuncts = 2
max_conjuncts = 1

# Print warnings or not?
ON = 1
OFF = 0
DISPLAY_WARNINGS = ON


""" 
"""


class partial_transition_function:
    def __init__(self, DNF, transition_matrix):
        self.b = DNF
        self.t = transition_matrix


def total_transition_function(A):
    return [partial_transition_function(np.array([0, 0, 0, 0, 0], ndmin=3), A)]


# Code Input
P_array = np.array([[[1, 0, 0, 0, 0]]])
B_array = np.array([[[1, 0, 0, -2, 6]]])
Q_array = np.array([[[1, 0, 0, 0, 6]]])
# T_function = [partial_transition_function(np.array([1,1,1,2,0] , ndmin = 3), np.array( [[1,2,3,1], [2,3,1,4] , [1,3,1,4], [0,0,0,1]] , ndmin = 2 )),
# partial_transition_function(np.array([1,1,1,-1,0], ndmin = 3), np.array( [[2,2,3,2], [2,3,2,4] , [2,3,3,4], [0,0,0,1]] , ndmin = 2 ) )]
T_function = total_transition_function(
    np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], ndmin=2))

""" Convert matrix to z3 expressions.
"""


def DNF_to_z3expr(D, p=''):
    if np.size(D) == 0:
        return True
    OP = {
        0: operator.eq,  # "=="
        -1: operator.le,  # "<="
        -2: operator.lt,  # "<"
        1: operator.ge,  # ">="
        2: operator.gt,  # ">"
    }
    d0 = len(D)
    d1 = len(D[0])
    d2 = len(D[0][0])
    return Or([
        And([
            OP[int(D[i][j][-2])](
                Sum([
                    D[i][j][k] * Int(('x%s'+p) % k)
                    for k in range(d2-2)
                ]),
                int(D[i][j][-1])
            )
            for j in range(d1)
        ])
        for i in range(d0)
    ])


def DNF_to_z3expr_prime(D):
    return DNF_to_z3expr(D, 'p')


# Testing the function:
P1 = np.array([[[1, 2, 3, 0, 0], [1, 2, 3, 0, -1]],
              [[1, 3, 3, 2, 1], [0, 0, 0, 0, 0]]])
S3 = DNF_to_z3expr(P1)
print(S3)  # To simplify expression, use simplify( .) function.


# A is a (n+1) * (n+1) matrix.
def trans_matrix_to_z3expr(A):
    d = len(A)
    return And([
        Int("x%sp" % i) ==
        Sum([
            int(A[i][j]) * Int("x%s" % j)
            for j in range(d-1)
        ]) +
        int(A[i][d-1])
        for i in range(d-1)
    ])


def trans_func_to_z3expr(f):
    ret = True
    for i in range(len(f)-1, -1, -1):
        ret = If(DNF_to_z3expr(f[i].b),
                 trans_matrix_to_z3expr(f[i].t),
                 ret)
    return ret

# Testing the functions.
# A = np.array([[1, 2, 3, 1], [2, 3, 1, 4], [1, 3, 1, 4], [0, 0, 0, 1]], ndmin=2)
# X = trans_matrix_to_z3expr(A)
# print(X, X)
# S = total_transition_function(A)
# print(S[0].b, S[0].t)


""" COC Extraction.
COC is short for coefficients, operators, constants
"""


def extract_COC_from_predicate(P):
    return (np.concatenate((P[0:N], P[N+1:])), P[N:N+1])


def extract_COC_from_conj_clause_internal(C):
    if np.size(C) == 0:
        return [np.empty(0), np.empty(0)]
    COC1 = extract_COC_from_predicate(C[0])
    COC2 = extract_COC_from_conj_clause_internal(C[1:])
    return [np.concatenate((COC1[0], COC2[0])), np.concatenate((COC1[1], COC2[1]))]


def extract_COC_from_conj_clause(C):
    A = extract_COC_from_conj_clause_internal(C)
    A[0] = (A[0].reshape(-1, N+1)).astype(int)
    A[1] = A[1].astype(int)
    return A


def get_predicate_from_COC(cc, o):
    return np.concatenate((np.concatenate((cc[0:N], np.array([o]))), cc[N:N+1]))


def get_DNF_from_COC_internal(CC, O):
    if np.size(O) == 0:
        return np.empty(0)
    A1 = get_predicate_from_COC(CC[0], O[0])
    A2 = get_DNF_from_COC_internal(CC[1:], O[1:])
    return np.concatenate((A1, A2))


def get_DNF_from_COC(COC):
    CC = COC[0]
    O = COC[1]
    A = get_DNF_from_COC_internal(CC, O)
    A = (A.reshape(-1, N+2)).astype(int)
    return A

# print(extract_COC_from_predicate(np.array([6,2,3,1,6]) ) )
# print(extract_COC_from_conj_clause( np.array([[1,2,3,1,3], [4,2,3,-1,6] , [7,3,3,-2,9]] ) ) )
# print(get_predicate_from_COC( np.array([1,2,3,3]) , 2) )
# print(get_DNF_from_COC( [np.array( [[1,2,3,3], [4,2,3,6], [7,2,3,9]] ), np.array([2,1,-1]) ] ) )
# print (get_DNF_from_COC(extract_COC_from_conj_clause( np.array([[1,1,1,-1,3], [4,5,3,-2,6] , [7,8,3,-1,9]] ))) )


# This assumes that T_function is a bijective function i.e. each partial transition matrix is non-singular, and that their respective domains are bijective (clearly too strong!)
# Also inverse matrix will not always have integer values!!!

def inverse_transition_function(f):
    if len(f) == 0:
        return []
    transition_matrix = f[0].t
    if (np.linalg.det(f[0].t)):
        inverse_partial_transition_matrix = np.linalg.inv(
            f[0].t).astype(int)  # Non-Singular Matrix
    else:
        if (DISPLAY_WARNINGS):
            print(
                "Transition function has a singular partial transition matrix, inverse isn't defined")
        # Singular Matrix, inverse doesn't exist!
        inverse_partial_transition_matrix = np.eye(N+1)
    DNF = f[0].b

    new_DNF = np.empty(0)
    for C in DNF:
        temp = extract_COC_from_conj_clause(C)
        temp[0] = np.dot(
            temp[0], inverse_partial_transition_matrix.transpose())
        new_CC = get_DNF_from_COC(temp)
        if (np.size(new_DNF) == 0):
            new_DNF = new_CC.reshape(1, -1, N+2)
        else:
            new_DNF = np.append(new_DNF, new_CC.reshape(1, -1, N+2), axis=0)
    return [partial_transition_function(new_DNF, inverse_partial_transition_matrix)] + inverse_transition_function(f[1:])


""" Constant Extraction
"""


def get_all_constants_in_2D_array(A):
    temp = A.tolist()
    rv = []
    for a in temp:
        rv = rv + a
    return list(set(rv))


def get_all_constants_in_3D_array(A):
    if len(A) == 0:
        return []
    return list(set(get_all_constants_in_2D_array(A[0]) + get_all_constants_in_3D_array(A[1:])))


def get_all_program_constants():
    rv = list(set(get_all_constants_in_3D_array(P_array) +
              get_all_constants_in_3D_array(B_array) + get_all_constants_in_3D_array(Q_array)))
    for partial in T_function:
        rv = list(set(rv + get_all_constants_in_3D_array(partial.b) +
                  get_all_constants_in_2D_array(partial.t)))
    return rv


programConstants = get_all_program_constants()


""" Normalization.
"""

# Let normalized versions only have <= or < operators.


def normalize_predicate(P):
    if (P[N] > 0):
        return np.array(np.multiply(P, -1), ndmin=2)
    elif (P[N] == 0):
        temp1, temp2 = np.multiply(P, -1), P
        temp1[N] = -1
        temp2[N] = -1
        return np.array([temp1, temp2], ndmin=2)
    return np.array(P, ndmin=2)


def normalize_conj_clause(C):
    if (len(C) == 0):
        return np.empty(shape=(0, N+2), dtype=int)
    return np.concatenate((normalize_predicate(C[0]), normalize_conj_clause(C[1:])))


def normalize_DNF_internal(D, conjunct_size):
    if (len(D) == 0):
        return np.empty(shape=(0, conjunct_size, N+2), dtype=int)
    C = normalize_conj_clause(D[0])
    padding_predicate = np.array([0, 0, 0, -1, 0], ndmin=1)
    while (len(C) < conjunct_size):
        C = np.concatenate((C, np.array(padding_predicate, ndmin=2)))
    return np.concatenate((np.array(C, ndmin=3), normalize_DNF_internal(D[1:], conjunct_size)))


def normalize_DNF(D):
    max_size = 0
    for C in D:
        curr_size = 0
        for P in C:
            curr_size = curr_size + (2 if (P[N] == 0) else 1)
        max_size = max(max_size, curr_size)
    return operator_normalize_DNF_internal(D, max_size)


# print(operator_normalize_predicate(np.array([1,2,0,-1,4])))
# print(operator_normalize_conjunctiveClause( np.array( [ [1,2,3,1,10], [1,3,1,0,0] , [1,3,1,-2,0] ] , ndmin = 2)) ) 
# print(operator_normalize_DNF(np.array( [ [[1,0,0,0,1] , [0,1,0,0,1] ], [[0,0,1,0,1] , [0,0,0,-2,0]] , [[0,0,1,-1,0] , [0,0,0,1,0]] ]) )  )
''' ********************************************************************************************************************'''


''' ********************************************************************************************************************'''
def random_guess_predicate (coefficient_domain, constant_domain, operator_probability_matrix):
    P = np.zeros(shape=n+2, dtype = int)
    for i in range(n):
        P[i] = np.random.choice( coefficient_domain )
    P[n] = np.random.choice([-2, -1, 0, 1, 2], p=operator_probability_matrix)
    P[n+1] = np.random.choice( constant_domain)
    return P #It doesn't matter if all coefficients of P are zero, then the predicate represents either True or False depending on the constant and operator value.

def random_guess_octagonalpredicate (coefficient_domain, constant_domain, operator_probability_matrix):
    P = np.zeros(shape=n+2, dtype = int)
    nonzero_coefficient_position = np.random.randint(0, n, size = (2)) 
    for i in range(n):
        if i in nonzero_coefficient_position:
            P[i] = np.random.choice( coefficient_domain )
        else:
            P[i] = 0
    P[n] = np.random.choice([-2, -1, 0, 1, 2], p=operator_probability_matrix)
    P[n+1] = np.random.choice( constant_domain)
    return P #It doesn't matter if all coefficients of P are zero, then the predicate represents either True or False depending on the constant and operator value. 



def random_guess_conjunctive_clause (coefficient_domain, constant_domain, conjunctive_clause_size, operator_probability_matrix, is_octagonal_predicate):
    if conjunctive_clause_size == 0:
        return np.empty(shape=(0, n+2), dtype = int)
    if (is_octagonal_predicate == 0):
        return np.append( np.array( random_guess_predicate(coefficient_domain, constant_domain, operator_probability_matrix) , ndmin = 2), 
                                    random_guess_conjunctive_clause(coefficient_domain, constant_domain, conjunctive_clause_size-1, operator_probability_matrix, is_octagonal_predicate)
                                                , axis = 0 )
    else:
        return np.append( np.array( random_guess_octagonalpredicate(coefficient_domain, constant_domain, operator_probability_matrix) , ndmin = 2), 
                                    random_guess_conjunctive_clause(coefficient_domain, constant_domain, conjunctive_clause_size-1, operator_probability_matrix, is_octagonal_predicate)
                                                , axis = 0 )        


def size_normalize_conjunctive_clause (conjunctive_clause, max_conjunctive_clause_size):
    if (conjunctive_clause.shape[0] == max_conjunctive_clause_size ):
        return conjunctive_clause
    return size_normalize_conjunctive_clause( np.append(conjunctive_clause, np.array([0,0,0,-1,0], ndmin = 2) , axis = 0), max_conjunctive_clause_size )



def random_guess_DNF(coefficient_domain, constant_domain, max_conjunctive_clause_size, number_of_disjuncts, operator_probability_matrix, is_octagonal_predicate):
    if (number_of_disjuncts == 0):
        return np.empty(shape = (0,max_conjunctive_clause_size,n+2), dtype = int)
    conjunctive_clause_size = np.random.randint(1, max_conjunctive_clause_size + 1 )
    return np.append( np.array(size_normalize_conjunctive_clause( random_guess_conjunctive_clause(coefficient_domain, constant_domain, conjunctive_clause_size, operator_probability_matrix
                            , is_octagonal_predicate), max_conjunctive_clause_size) , ndmin =3) , random_guess_DNF(coefficient_domain, constant_domain, 
                            max_conjunctive_clause_size, number_of_disjuncts - 1, operator_probability_matrix, is_octagonal_predicate), axis = 0  )



def random_guess_invariant (coefficient_domain, constant_domain, max_conjunctive_clause_size, max_number_of_disjuncts, operator_probability_matrix, is_octagonal_predicate):
    number_of_disjuncts = np.random.randint(1, max_number_of_disjuncts + 1 )
    return random_guess_DNF(coefficient_domain, constant_domain, max_conjunctive_clause_size, number_of_disjuncts, operator_probability_matrix, is_octagonal_predicate)

def get_GP_list (a, r, N):
    rv = []
    curr = 1.0
    for i in range(1, N + 1):
        if (i == 1):
            curr = a
        else:
            curr = curr * r
        rv = rv + [curr]
    return rv

# print(get_GP_list(0.2,0.5,10))

# Assumes negative_error is a negative value
def normalize_negative_error_in_list( approximated_list, negative_error):
    if ( negative_error >= 0):
        return approximated_list
    elif (approximated_list == []):
        if(DISPLAY_WARNINGS == ON):
            print("Too much negative error in generating this list!")
        return [negative_error]
    return [max(0, approximated_list[0] + negative_error) ] + normalize_negative_error_in_list( approximated_list[1:] , negative_error + approximated_list[0]  )

# This function assumes that domainList is sorted, and elements cannot be repeated
def get_geometric_probability_list(domainList, centre, r):
    index = domainList.index(centre)
    s1 = index
    s2 = len(domainList) - index - 1
    
    sublist1 = []
    sublist2 = []
    a = 1.0
    if (s1 == 0):
        a =  ((float) (1 - r)) / (1 - r**s2)
        sublist1 = []
        sublist2 = get_GP_list(a, r, s2)
    elif (s2 == 0):
        a =  ((float) (1 - r)) / (1 - r**s1)
        sublist1 = get_GP_list(a, r, s1)[::-1]
        sublist2 = get_GP_list(a, r, s2)
    else:
        a =  ((float) (1 - r)**2) / (1 - r**s1) * (1 - r**s2)
        sublist1 = get_GP_list(a, r, s1)[::-1]
        sublist2 = get_GP_list(a, r, s2)
    
    probabilityList = sublist1 + [ 0 ] + sublist2
    
    computation_error = 1 - sum(probabilityList)
    if(computation_error >= 0):
        if (index == 0):
            probabilityList[index + 1] = probabilityList[index + 1] + (computation_error)
        elif (index == len(probabilityList) - 1):
            probabilityList[index - 1] = probabilityList[index - 1] + (computation_error)
        else:
            probabilityList[index - 1] = probabilityList[index - 1] + (computation_error * 0.5)
            probabilityList[index + 1] = probabilityList[index + 1] + (computation_error * 0.5)
    else:
        probabilityList = normalize_negative_error_in_list(probabilityList[0:index], 0.5 * computation_error) + probabilityList[index: index + 1] + (normalize_negative_error_in_list( (probabilityList[index + 1:])[::-1], 0.5 * computation_error ))[::-1]
    return probabilityList

# Implement the is_octagonal_predicate version too!
# Assumes I_prev has correct numpy dimensions.
def MC_guess_invariant(I_prev, coefficient_domain, constant_domain, max_number_of_disjuncts, change_size_probability, change_value_probability_GPratio,  is_octagonal_predicate, 
                                                                                                                                operator_probability_matrix):
    # Need sorted lists
    coefficient_domain.sort()
    operator_domain = [-2,-1,0,1,2]
    constant_domain.sort()

    prev_number_of_disjuncts = I_prev.shape[0]

    if( np.random.choice([0,1], p=np.array([1 - change_size_probability, change_size_probability])) == 1   ):   
        domain_of_disjunct_values = (list((range(1, max_number_of_disjuncts + 1))))
        domain_of_disjunct_values.remove(prev_number_of_disjuncts)
        if not domain_of_disjunct_values:
            if (DISPLAY_WARNINGS == ON):
                print("Size change not possible becuase domain of disjunct values is empty,")
            return I_prev
        new_number_of_disjuncts = np.random.choice( domain_of_disjunct_values ) 
        if (new_number_of_disjuncts < prev_number_of_disjuncts):
            return I_prev[0:new_number_of_disjuncts]
        else:
            additional_disjuncts = new_number_of_disjuncts - prev_number_of_disjuncts
            max_conjunctive_clause_size = I_prev.shape[1]
            return np.append(I_prev, random_guess_DNF(coefficient_domain, constant_domain, max_conjunctive_clause_size, 
                        additional_disjuncts, operator_probability_matrix, is_octagonal_predicate), axis = 0)
    else: #size is constant:
        prev_CC_position = np.random.choice( range(0, prev_number_of_disjuncts) )
        prev_CC = I_prev[prev_CC_position]
        prev_CC_size = prev_CC.shape[0]
        prev_P_position = np.random.choice(  range(0, prev_CC_size) )
        prev_P = prev_CC[prev_P_position]
        index_to_change = np.random.choice( range(n+2) )
        new_P = prev_P
        if (is_octagonal_predicate == 0):
            if (index_to_change < n):
                domain = coefficient_domain
            elif (index_to_change == n):
                domain = operator_domain
            else:
                domain = constant_domain
            prob_list = get_geometric_probability_list(domain, prev_P[index_to_change], change_value_probability_GPratio)
            newval = np.random.choice( domain, p= prob_list )
            new_P[index_to_change] = newval        
        else:
            if (index_to_change < n):
                domain = coefficient_domain
                prob_list = get_geometric_probability_list(domain, prev_P[index_to_change], change_value_probability_GPratio)
                newval = np.random.choice( domain, p= prob_list )
                new_P[index_to_change] = newval                   
                nonzero_index = (np.transpose(np.nonzero(prev_P[0:n])).reshape(-1)).tolist()
                if (len(nonzero_index) == 2 and index_to_change not in nonzero_index):
                    index_to_reset = np.random.choice(nonzero_index)
                    new_P[index_to_reset] = 0
            else:
                if (index_to_change == n):
                    domain = operator_domain
                else:
                    domain = constant_domain
                prob_list = get_geometric_probability_list(domain, prev_P[index_to_change], change_value_probability_GPratio)
                newval = np.random.choice( domain, p= prob_list )
                new_P[index_to_change] = newval   
        new_CC = prev_CC
        new_CC[prev_P_position] = new_P
        I = I_prev
        I[prev_CC_position] = new_CC
        return I


def guess_invariant_smallConstants (maxConstantValue, max_conjunctive_clause_size, max_number_of_disjuncts, operator_probability_matrix):
    smallConstants_domain = range(-maxConstantValue, maxConstantValue + 1)
    return random_guess_invariant(smallConstants_domain, smallConstants_domain, max_conjunctive_clause_size, max_number_of_disjuncts, operator_probability_matrix, 0)

def guess_invariant_octagonaldomain ( listOfConstants, max_conjunctive_clause_size, max_number_of_disjuncts, operator_probability_matrix):
    return random_guess_invariant( [-1,0,1], listOfConstants, max_conjunctive_clause_size, max_number_of_disjuncts, operator_probability_matrix, 1  )

def guess_invariant_octagonaldomain_extended ( listOfConstants, max_conjunctive_clause_size, max_number_of_disjuncts, operator_probability_matrix):
    return random_guess_invariant( [-1,0,1], listOfConstants, max_conjunctive_clause_size, max_number_of_disjuncts, operator_probability_matrix, 0  )

def guess_invariant_nearProgramConstants ( values_in_program, k,  max_conjunctive_clause_size, max_number_of_disjuncts, operator_probability_matrix):
    list_of_values=[]
    for value in values_in_program:
        list_of_values = list_of_values + list(range(value - k, value + k + 1))
    list_of_values = list(set(list_of_values)) #remove duplicates
    return random_guess_invariant( list_of_values, list_of_values, max_conjunctive_clause_size, max_number_of_disjuncts, operator_probability_matrix, 0)


def MC_guess_invariant_smallConstants (I_prev, maxConstantValue, max_number_of_disjuncts, change_size_probability, change_value_probability_GPratio, operator_probability_matrix):
    smallConstants_domain = list(range(-maxConstantValue, maxConstantValue + 1))
    return MC_guess_invariant(I_prev, smallConstants_domain, smallConstants_domain , max_number_of_disjuncts, change_size_probability, 
                            change_value_probability_GPratio, 0, operator_probability_matrix )



def MC_guess_invariant_octagonaldomain (I_prev, listOfConstants, max_number_of_disjuncts, change_size_probability, change_value_probability_GPratio, operator_probability_matrix):
    return MC_guess_invariant(I_prev, [-1,0,1], listOfConstants , max_number_of_disjuncts, change_size_probability, 
                            change_value_probability_GPratio, 1, operator_probability_matrix )    

def MC_guess_invariant_octagonaldomain_extended (I_prev, listOfConstants, max_number_of_disjuncts, change_size_probability, change_value_probability_GPratio, operator_probability_matrix):
    return MC_guess_invariant(I_prev, [-1,0,1], listOfConstants , max_number_of_disjuncts, change_size_probability, 
                            change_value_probability_GPratio, 0, operator_probability_matrix )  

def MC_guess_invariant_nearProgramConstants (I_prev, values_in_program, k, max_number_of_disjuncts, change_size_probability, change_value_probability_GPratio, operator_probability_matrix):
    list_of_values=[]
    for value in values_in_program:
        list_of_values = list_of_values + list(range(value - k, value + k + 1))
    list_of_values = list(set(list_of_values)) #remove duplicates
    return MC_guess_invariant(I_prev,list_of_values, list_of_values , max_number_of_disjuncts, change_size_probability, 
                            change_value_probability_GPratio, 0, operator_probability_matrix )  

# print(random_guess_predicate([-1,1,0] , programConstants, np.array([0.2, 0.2, 0.2, 0.2, 0.2])) )
# print(random_guess_octagonalpredicate([-1,1,0] , programConstants, np.array([0.2, 0.2, 0.2, 0.2, 0.2])) )
# print(random_guess_conjunctive_clause([-1,1,0] , programConstants, 3,  np.array([0.2, 0.2, 0.2, 0.2, 0.2]), 0) )
# print(size_normalize_conjunctive_clause(random_guess_conjunctive_clause([-1,1,0] , programConstants, 3,  np.array([0.2, 0.2, 0.2, 0.2, 0.2]), 1), 5 ) )
# print( random_guess_DNF([-1,1,0] , programConstants, 3, 4, np.array([0.2, 0.2, 0.2, 0.2, 0.2]), 1) )
# print( random_guess_invariant([-1,1,0] , programConstants, 3, 4, np.array([0.2, 0.2, 0.2, 0.2, 0.2]), 1) )
# print(guess_invariant_smallConstants(10, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) ))
# print(guess_invariant_octagonaldomain(programConstants, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) ))
# print(guess_invariant_octagonaldomain_extended(programConstants, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) ))
# print(guess_invariant_nearProgramConstants(programConstants, 2, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) ))

# Testing the MC functions:
# I_p = guess_invariant_octagonaldomain(programConstants, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) )
# print(I_p)
# print(MC_guess_invariant(I_p, [-1,0,1], programConstants, 3, 0.1, 0.5, 1, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) ))

# I_1 = guess_invariant_smallConstants(10, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) )
# print(I_1)
# print(MC_guess_invariant_smallConstants (I_1, 10, 3, 0.1, 0.5, np.array([0.2, 0.2, 0.2, 0.2, 0.2])))

# I_2 = guess_invariant_octagonaldomain(programConstants, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) )
# print(I_2)
# print(MC_guess_invariant_octagonaldomain (I_2, programConstants, 3, 0.1, 0.5, np.array([0.2, 0.2, 0.2, 0.2, 0.2])))

# I_3 = guess_invariant_octagonaldomain_extended(programConstants, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) )
# print(I_3)
# print(MC_guess_invariant_octagonaldomain_extended (I_3, programConstants, 3, 0.1, 0.5, np.array([0.2, 0.2, 0.2, 0.2, 0.2])))

# I_4 = guess_invariant_nearProgramConstants(programConstants, 5, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) )
# print(I_4)
# print(MC_guess_invariant_nearProgramConstants(I_4, programConstants, 5, 3, 0.1, 0.5, np.array([0.2, 0.2, 0.2, 0.2, 0.2])))
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

""" 
Printing Utility.
"""


def print_DNF(D):
    print(DNF_to_z3expr(D))

# Testing
# print_predicate(np.array([1,2,3,0,0], ndmin = 1))
# print_conjunctiveClause(np.array( [[1,2,3,0,0], [1,3,4,1,2] , [1,1,3,-2,3]] , ndmin = 2))
# print_DNF(np.array( [[[1,2,3,0,0], [1,1,3,1,2] , [1,2,3,-2,3]] , [[1,1,2,0,43], [1,1,1,-1,2] , [1,1,2,2,3]]] , ndmin = 3), 0)


""" Counter Example Generation.
"""

P = DNF_to_z3expr(P_array)
B = DNF_to_z3expr(B_array)
Q = DNF_to_z3expr(Q_array)
T_inv = inverse_transition_function(T_function)
T = trans_func_to_z3expr(T_function)


def C1(I):
    return Implies(P, I)


def C2(I, I_prime):
    return Implies(And(B, I, T), I_prime)


def C3(I):
    return Implies(And(I, Not(B)), Q)


def System(I):
    return And(C1(I), C2(I), C3(I))


""" Get a list of counterexamples.
"""


def get_cex(C, num_cex):
    result = []
    s = Solver()
    s.add(Not(C))
    while len(result) < num_cex and s.check() == sat:
        m = s.model()
        result.append(m)
        # Create a new constraint the blocks the current model
        block = []
        for d in m:
            # d is a declaration
            if d.arity() > 0:
                raise Z3Exception("uninterpreted functions are not supported")
            # create a constant from declaration
            c = d()
            if is_array(c) or c.sort().kind() == Z3_UNINTERPRETED_SORT:
                raise Z3Exception(
                    "arrays and uninterpreted sorts are not supported")
            block.append(c != m[d])
        s.add(Or(block))
    else:
        if len(result) < num_cex and s.check() != unsat:
            print("Solver can't verify or disprove")
            return result
    return result


def get_cex_C1(I, number_of_cex):
    return get_cex(C1(I), number_of_cex)


def get_cex_C2(I, I_prime, number_of_cex):
    return get_cex(C2(I, I_prime), number_of_cex)


def get_cex_C3(I, number_of_cex):
    return get_cex(C3(I), number_of_cex)

# I_g_array = guess_small_constants(10, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) )
# I_g = DNF_to_z3expr(I_g_array)
# print("--------------------------------")
# print(I_g)
# print("--------------------------------")
# print(get_cex_C1(I_g, 10) ,'\n')
# print("--------------------------------")
# print(get_cex_C2(I_g, 10), '\n')
# print("--------------------------------")
# print(get_cex_C3(I_g, 10), '\n')


""" Distance Functions.
"""


def dist(x, p):
    return np.linalg.norm(x - p)

def distance_point_conjunctiveClause(p, C):
    C = operator_normalize_conjunctiveClause(C)
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
        d = min(d, distance_point_conj_clauses(p, C))
    return d

# Testing:
# print(distance_point_conj_clauses( np.array( [-3,1,3], ndmin = 1), np.array( [ [1,2,3,1,10], [1,3,1,0,0] ] , ndmin = 2)) )
# print(distance_point_DNF( np.array( [-3,3,1], ndmin = 1), np.array( [ [ [-7,1,3,-2,3], [1,2,1,0,2], [3,1,3, 1, 4] ], [ [1,3,1,1,10], [1,3,1,0,0], [0,0,0,0,0] ] ]    )) )


""" Sampling Functions.
"""

# This is specific for this clause system.


def sample_points_from_DNF(D, number_of_points, sample_list):
    if (number_of_points == 0):
        return sample_list

    s = Solver()
    s.add(Implies(True, D(x, y, z)))
    r = s.check()
    output = r.__repr__()
    if output == "sat":
        sample_point = s.model()
        sample_list.append(sample_point)
        return sample_points_from_DNF(lambda u, v, w: And(D(u, v, w), Or(u != sample_point.evaluate(x), v != sample_point.evaluate(y), w != sample_point.evaluate(z))), number_of_points - 1, sample_list)
    elif output == "unsat":
        return sample_list
    else:
        print("Sampler can't sample, it says: %s" % (r))
        return sample_list

# Assumes transition function is a total function.


def unroll_chain_from_starting_point(pt_matrix, transition_function, conditional_predicate, number_of_points, sample_list):
    if (number_of_points == 0):
        return sample_list
    if (simplify(conditional_predicate(int(pt_matrix[0]), int(pt_matrix[1]), int(pt_matrix[2])))):
        sample_list.append(pt_matrix[0:n])
        new_pt_matrix = np.empty(n + 1, int)
        for partial_tf in transition_function:
            if (simplify(DNF_to_z3expr(partial_tf.b)(int(pt_matrix[0]), int(pt_matrix[1]), int(pt_matrix[2])))):
                new_pt_matrix[:] = np.dot(
                    pt_matrix, np.transpose(partial_tf.t))
        return unroll_chain_from_starting_point(new_pt_matrix, transition_function, conditional_predicate, number_of_points - 1, sample_list)
    else:
        return sample_list


def get_positive_points(sampling_breadth, sampling_depth):
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
        unroll_chain_from_starting_point(
            pt_matrix, T_function, B, SD + 1, list_of_positive_points)

    uniques = []  # remove duplicates
    for arr in list_of_positive_points:
        if not any(np.array_equal(arr, unique_arr) for unique_arr in uniques):
            uniques.append(arr)
    return uniques


def get_negative_points(sampling_breadth, sampling_depth):
    temp_list = []
    sample_points_from_DNF(lambda x, y, z: And(
        Not(Q(x, y, z)), Not(B(x, y, z))), sampling_breadth, temp_list)
    breadth_list_of_negative_points = []
    for sample in temp_list:
        pt_x = sample.evaluate(x).as_long()
        pt_y = sample.evaluate(y).as_long()
        pt_z = sample.evaluate(z).as_long()
        breadth_list_of_negative_points.append(np.array([pt_x, pt_y, pt_z]))
    list_of_negative_points = []
    for pt in breadth_list_of_negative_points:
        pt_matrix = np.concatenate((pt, np.array([1])))
        unroll_chain_from_starting_point(pt_matrix, T_inv, lambda x, y, z: Or(
            Not(Q(x, y, z)), B(x, y, z)), SD + 1, list_of_negative_points)

    uniques = []  # remove duplicates
    for arr in list_of_negative_points:
        if not any(np.array_equal(arr, unique_arr) for unique_arr in uniques):
            uniques.append(arr)
    return uniques

# Testing these functions
# print(get_positive_points(SB, SD))
# print(get_negative_points(SB, SD))


""" Cost Functions.
"""


def J1(I, cex_list, n):
    """
    A cost function.

    :param I: I as a numpy array
    :param cex_list: a list of counterexamples
    :param n: number of variables
    :return: the cost
    """
    error = 0
    for cex in cex_list:
        pt = [cex.evaluate(Int("x%i"), model_completion=True).as_long()
              for i in range(n)]
        point = np.array(pt)
        error = max(error, distance_point_DNF(point, I))
    return error + len(cex_list)

# Traditionally try to 'guess' which cex are supposed to be negative, and which are supposed to be positive, and then there is a relative ratio; but we skip that here.


def J2(cex_list):
    return len(cex_list)


def J3(Q: np.ndarray, cex_list, n):
    """
    A cost function.

    :param Q: Q as a numpy array
    :param cex_list: a list of counterexamples
    :param n: number of variables
    :return: the cost
    """
    error = 0
    for cex in cex_list:
        pt = [cex.evaluate(Int("x%i"), model_completion=True).as_long()
              for i in range(n)]
        point = np.array(pt)
        error = max(error, distance_point_DNF(point, Q))
    return error + len(cex_list)

# Testing these functions
# I_g_array = guess_small_constants(10, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) )
# I_g = convert_DNF_to_lambda(I_g_array)
# cex_list1 = get_cex_C1(I_g, 10)
# cex_list2 = get_cex_C2(I_g, 10)
# cex_list3 = get_cex_C3(I_g, 10)
# print(J1(I_g_array, cex_list1))
# print(J2(cex_list2))
# print(J3(Q_array, cex_list3))


""" Main function.
"""

# Here in all our guess strategies, we assume that conjunctive clause size is same.

'''
guess_strategy code:
(1,k) -> smallConstants(k)
(2,0) -> octagonaldomain 
(2,1) -> octagonaldomain_extended
(3,k) -> nearProgramConstants(k)
'''


def guess_invariant(guesses, guess_strategy, no_of_conjuncts, no_of_disjuncts):
    cost = float('inf')
    count = 0
    I_g = np.empty( shape = (0, no_of_conjuncts, n+2))
    while (cost != 0 and count < guesses):
        count += 1
        if (guess_strategy[0] == 1):
            I_g_array = guess_small_constants(
                guess_strategy[1], no_of_conjuncts, no_of_disjuncts, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
        elif (guess_strategy[0] == 2):
            if (guess_strategy[1] == 0):
                I_g_array = guess_octagonaldomain(
                    programConstants, no_of_conjuncts, no_of_disjuncts, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
            else:
                I_g_array = guess_octagonaldomain_extended(
                    programConstants, no_of_conjuncts, no_of_disjuncts, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
        elif (guess_strategy[0] == 3):
            I_g_array = guess_near_prog_const(
                programConstants, guess_strategy[1], no_of_conjuncts, no_of_disjuncts,  np.array([0.2, 0.2, 0.2, 0.2, 0.2]))

        I_g = DNF_to_z3expr(I_g_array)
        I_g_prime = DNF_to_z3expr_prime(I_g_array)

        C1_cex_list = get_cex_C1(I_g, s)
        C2_cex_list = get_cex_C2(I_g, I_g_prime, s)
        C3_cex_list = get_cex_C3(I_g, s)

        # Get costFunction
        cost1 = J1(I_g_array, C1_cex_list, N)
        cost2 = J2(C2_cex_list)
        cost3 = J3(Q_array, C3_cex_list, N)
        cost = K1*cost1 + K2*cost2 + K3*cost3

        # print(cost1, cost2, cost3, end = '')
        print('   ', round(cost, 2),'\n')
    return (I_g_array, cost)

def MC_invariant_guess (guesses, guess_strategy, no_of_conjuncts , no_of_disjuncts, k , r):
    count = 0
    curr_cost = float('inf')
    curr_I = np.empty(shape = (0,max_conjuncts, n+2))
    while (curr_cost != 0 and count < guesses):
        count = count + 1
        prev_cost = curr_cost
        prev_I = curr_I
        if (count == 1):
            rv = random_invariant_guess(1, guess_strategy, no_of_conjuncts , no_of_disjuncts)
            curr_cost = rv[1]
            curr_I = rv[0]
            continue
        else:
            if (guess_strategy[0] == 1):
                curr_I = MC_guess_invariant_smallConstants (prev_I, guess_strategy[1], no_of_disjuncts, k, r, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
            elif (guess_strategy[0] == 2):
                if (guess_strategy[1] == 0):
                    curr_I = MC_guess_invariant_octagonaldomain (prev_I, programConstants, no_of_disjuncts, k, r, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
                else:
                    curr_I  = MC_guess_invariant_octagonaldomain_extended (prev_I, programConstants, no_of_disjuncts, k, r, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
            elif (guess_strategy[0] == 3):
                curr_I = MC_guess_invariant_nearProgramConstants(prev_I, programConstants, guess_strategy[1],no_of_disjuncts, k, r, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))     
            
            curr_I_g = convert_DNF_to_lambda(curr_I)
            C1_cexList = GenerateCexList_C1 (curr_I_g , s)
            # print(C1_cexList)
            C2_cexList = GenerateCexList_C2 ( curr_I_g , s)
            # print(C2_cexList)
            C3_cexList = GenerateCexList_C3 ( curr_I_g , s)
            # print(C3_cexList)

            # Get costFunction
            cost1 = J1(curr_I, C1_cexList)
            cost2 = J2(C2_cexList)
            cost3 = J3(Q_array, C3_cexList)
            curr_cost = K1*cost1 + K2*cost2 + K3*cost3

            if (curr_cost > prev_cost):
                curr_cost = prev_cost
                curr_I = prev_I 
                continue
            else:
                prob_of_change = min(1.0 - (curr_cost/prev_cost) + 0.05, 1.0) #0.05 is added so that prob_of_change is not zero if curr_cost = prev_cost
                if( np.random.choice([0,1], p=np.array([1 - prob_of_change, prob_of_change])) == 0   ): #Don't change
                    curr_cost = prev_cost
                    curr_I = prev_I #Isn't it pointer assignment here? Won't this give a really bad error, the same as the semantic bug as before?!?
                else:
                    print(count,'   ', end = '')
                    print_DNF(curr_I, 0)
                    print("\t", end = '')        
                    # print(cost1, cost2, cost3, end = '')
                    print('   ', round(curr_cost, 2),'\n')
    return





# random_invariant_guess(max_guesses, (2,0), max_conjuncts, max_disjuncts )
MC_invariant_guess (max_guesses, (3,1), max_conjuncts, max_disjuncts, 0.1 , 0.5)
