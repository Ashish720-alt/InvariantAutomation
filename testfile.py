from z3 import *

import numpy as np
from scipy.optimize import minimize, LinearConstraint

# Number of variables
n = 3

# Print warnings or not?
ON = 1
OFF = 0
DISPLAY_WARNINGS = ON

class partial_transition_function:
    def __init__(self, DNF , transition_matrix):
        self.b = DNF
        self.t = transition_matrix

def total_transition_function (A):
    return [ partial_transition_function(np.array([0,0,0,0,0], ndmin = 3), A ) ]


# Code Input
P_array = np.array( [ [[1,0,0,0,0] ] ])
B_array = np.array( [ [ [1,0,0,-2,6]  ] ] )
Q_array = np.array( [ [[1,0,0,0,6] ] ] )
# T_function = [partial_transition_function(np.array([1,1,1,2,0] , ndmin = 3), np.array( [[1,2,3,1], [2,3,1,4] , [1,3,1,4], [0,0,0,1]] , ndmin = 2 )), 
    # partial_transition_function(np.array([1,1,1,-1,0], ndmin = 3), np.array( [[2,2,3,2], [2,3,2,4] , [2,3,3,4], [0,0,0,1]] , ndmin = 2 ) )] 
T_function = total_transition_function(np.array( [[1,0,0,1], [0,1,0,0], [0,0,1,0], [0,0,0,1]] , ndmin = 2 ))


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
        print(prev_CC)
        prev_CC_size = prev_CC.shape[0]
        prev_P_position = np.random.choice(  range(0, prev_CC_size) )
        prev_P = prev_CC[prev_P_position]
        print(prev_P)
        index_to_change = np.random.choice( range(n+2) )

        new_P = prev_P
        if (index_to_change < n):
            domain = coefficient_domain
        elif (index_to_change == n):
            domain = operator_domain
        else:
            domain = constant_domain
        
        prob_list = get_geometric_probability_list(domain, prev_P[index_to_change], change_value_probability_GPratio)
        print(prev_CC_position, prev_P_position, index_to_change )
        print(domain, prev_P, prev_P[index_to_change], change_value_probability_GPratio, prob_list)
        newval = np.random.choice( domain, p= prob_list )
        new_P[index_to_change] = newval        
        new_CC = prev_CC
        new_CC[prev_P_position] = new_P
        I = I_prev
        I[prev_CC_position] = new_CC
        # print(prev_P[index_to_change], domain, prob_list,  newval )
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

programConstants.sort()
# print([-1,0,1], programConstants)
I_p = guess_invariant_octagonaldomain_extended(programConstants, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) )
print("old:\n", I_p,'\n', "new:\n", MC_guess_invariant(I_p, [-1,0,1], programConstants, 3, 0.1, 0.5, 0, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) ))
''' ********************************************************************************************************************'''