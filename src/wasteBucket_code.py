from z3 import *
import numpy as np
from scipy.optimize import minimize, LinearConstraint

''' The code of some functions here contains some theory which I haven't written on paper yet.'''

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

# Number of variables
n = 3

# Only for 3 variable case
x, y, z, xp, yp, zp = Ints('x y z xp yp zp')


class partial_transition_function:
    def __init__(self, DNF, transition_matrix):
        self.b = DNF
        self.t = transition_matrix


def total_transition_function(A):
    return [partial_transition_function(np.array([0, 0, 0, 0, 0], ndmin=3), A)]


''' ********************************************************************************************************************'''
# 'COC' is short for coefficients, operators, constants


def extract_COC_from_predicate(P):
    return (np.concatenate((P[0:n], P[n+1:])), P[n:n+1])


def extract_COC_from_conjunctiveClause_internal(C):
    if np.size(C) == 0:
        return [np.empty(0), np.empty(0)]
    COC1 = extract_COC_from_predicate(C[0])
    COC2 = extract_COC_from_conjunctiveClause_internal(C[1:])
    return [np.concatenate((COC1[0], COC2[0])), np.concatenate((COC1[1], COC2[1]))]


def extract_COC_from_conjunctiveClause(C):
    A = extract_COC_from_conjunctiveClause_internal(C)
    A[0] = (A[0].reshape(-1, n+1)).astype(int)
    A[1] = A[1].astype(int)
    return A


def get_predicate_from_COC(cc, o):
    return np.concatenate((np.concatenate((cc[0:n], np.array([o]))), cc[n:n+1]))


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
    A = (A.reshape(-1, n+2)).astype(int)
    return A

# print(extract_COC_from_predicate(np.array([6,2,3,1,6]) ) )
# print(extract_COC_from_conjunctiveClause( np.array([[1,2,3,1,3], [4,2,3,-1,6] , [7,3,3,-2,9]] ) ) )
# print(get_predicate_from_COC( np.array([1,2,3,3]) , 2) )
# print(get_DNF_from_COC( [np.array( [[1,2,3,3], [4,2,3,6], [7,2,3,9]] ), np.array([2,1,-1]) ] ) )
# print (get_DNF_from_COC(extract_COC_from_conjunctiveClause( np.array([[1,1,1,-1,3], [4,5,3,-2,6] , [7,8,3,-1,9]] ))) )


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
        inverse_partial_transition_matrix = np.eye(n+1)
    DNF = f[0].b

    new_DNF = np.empty(0)
    for C in DNF:
        temp = extract_COC_from_conjunctiveClause(C)
        temp[0] = np.dot(
            temp[0], inverse_partial_transition_matrix.transpose())
        new_CC = get_DNF_from_COC(temp)
        if (np.size(new_DNF) == 0):
            new_DNF = new_CC.reshape(1, -1, n+2)
        else:
            new_DNF = np.append(new_DNF, new_CC.reshape(1, -1, n+2), axis=0)
    return [partial_transition_function(new_DNF, inverse_partial_transition_matrix)] + inverse_transition_function(f[1:])


''' ********************************************************************************************************************'''

''' ********************************************************************************************************************'''
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
            if (simplify(convert_DNF_to_lambda(partial_tf.b)(int(pt_matrix[0]), int(pt_matrix[1]), int(pt_matrix[2])))):
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


''' ********************************************************************************************************************'''
