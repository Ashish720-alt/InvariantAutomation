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



'repr.py:'

def I(n):
    return np.identity(n + 1, dtype=int)

def E(n, pos ): #Indices run from 1 to n+1
    T = np.zeros(shape=(n+1, n+1), dtype=int)
    T[pos[0]-1][pos[1]-1] = 1
    return T



class PartialTransitionFunc:
    def __init__(self, DNF, transition_matrix):
        self.b = DNF
        self.t = transition_matrix


def TotalTransitionFunc(*args):
    return [PartialTransitionFunc(x[0], x[1]) for x in args]


def SimpleTotalTransitionFunc(A):
    # len(A[0]) is n+1, reqd is n+2
    return [ PartialTransitionFunc(np.zeros((1, 1, len(A[0]) + 1)), A) ]



class Repr:
    def __init__(self, P, B, Q, T):
        """ 
        TODO: T description.
        """
        self.num_var = len(P[0][0]) - 2  # n+1 is op, n+2 is const
        self.P = P.copy()
        self.P_z3expr = DNF_to_z3expr(P)
        self.B = B.copy()
        self.B_z3expr = DNF_to_z3expr(B)
        self.Q = Q.copy()
        self.Q_z3expr = DNF_to_z3expr(Q)
        self.T = T.copy()
        self.T_z3expr = trans_func_to_z3expr(self.T)

        def _extract_consts():
            def f(x): return set(x.flatten())
            ret = f(self.P) | f(self.B) | f(self.Q)
            for partial in self.T:
                ret |= f(partial.b) | f(partial.t)
            return list(ret) 
        self.consts = _extract_consts()

    def get_num_var(self):
        return self.num_var

    def get_consts(self):
        return self.consts

    def get_P(self):
        return self.P

    def get_B(self):
        return self.B

    def get_Q(self):
        return self.Q

    def get_T(self):
        return self.T

    def get_P_z3expr(self):
        return self.P_z3expr

    def get_B_z3expr(self):
        return self.B_z3expr

    def get_Q_z3expr(self):
        return self.Q_z3expr

    def get_T_z3expr(self):
        return self.T_z3expr


''' ********************************************************************************************************************'''

class GuessStrategy(Enum):
    SMALL_CONSTANT = auto()
    NEAR_CONSTANT = auto()
    OCTAGONAL_DOMAIN = auto()
    OCTAGONAL_DOMAIN_EXTENDED = auto()
    MC_SMALL_CONSTANT = auto()
    MC_NEAR_CONSTANT = auto()
    MC_OCTAGONAL_DOMAIN = auto()
    MC_OCTAGONAL_DOMAIN_EXTENDED = auto()

    @staticmethod
    def mc_to_not_mc(gs):
        __mapping = {
            GuessStrategy.MC_SMALL_CONSTANT: GuessStrategy.SMALL_CONSTANT,
            GuessStrategy.MC_NEAR_CONSTANT: GuessStrategy.NEAR_CONSTANT,
            GuessStrategy.MC_OCTAGONAL_DOMAIN: GuessStrategy.OCTAGONAL_DOMAIN,
            GuessStrategy.MC_OCTAGONAL_DOMAIN_EXTENDED: GuessStrategy.OCTAGONAL_DOMAIN_EXTENDED,
        }
        return __mapping[gs]


class Guess:
    def __init__(self, num_var, max_num_conj, max_num_disj, op_dist, strategy: GuessStrategy, **kwargs):
        """ 
        :num_var: number of variables. 
        :num_conj: number of conjuntives.
        :num_disj: number of disjuntives.
        :op_dist: a len=5 1d matrix, the prob of operators.
        :strategy: see GuessStrategy.
        :**kwargs: {
            SMALL_CONSTANT:
                max_const: the maximum value of constants.
            NEAR_CONSTANT:
                consts: constants in the programs.
                range: the range of adjustment.
            OCTAGONAL_DOMAIN:
            OCTAGONAL_DOMAIN_EXTENDED:
                consts: a list of constants.
            MC versions additional parameters:
                prev_I
                change_size_prob
                change_value_prob_ratio
        }

        Example:
            g = Guess(...)
            g.guess() # returns the new invariant
        """
        self.num_var = num_var
        self.len_pred = num_var + 2
        self.max_num_conj = max_num_conj
        self.max_num_disj = max_num_disj
        self.op_dist = op_dist
        assert(len(self.op_dist) == len(conf.OP_DOMAIN))

        self.NUM_OCT_NONZERO_POS = min(self.num_var, 2)

        self.__guess = None
        if (strategy == GuessStrategy.SMALL_CONSTANT):
            self.__guess = lambda: self.guess_inv_small_const(
                kwargs["max_const"])
        elif (strategy == GuessStrategy.NEAR_CONSTANT):
            self.__guess = lambda: self.guess_inv_near_const(
                kwargs["consts"], kwargs["range"])
        elif (strategy == GuessStrategy.OCTAGONAL_DOMAIN):
            self.__guess = lambda: self.guess_inv_oct(kwargs["consts"])
        elif (strategy == GuessStrategy.OCTAGONAL_DOMAIN_EXTENDED):
            self.__guess = lambda: self.guess_inv_oct_ext(kwargs["consts"])
        elif (strategy == GuessStrategy.MC_SMALL_CONSTANT):
            self.__guess = lambda: self.mc_guess_inv_small_const(
                kwargs["max_const"], kwargs["prev_I"], kwargs["change_size_prob"], kwargs["change_value_prob_ratio"])
        elif (strategy == GuessStrategy.MC_NEAR_CONSTANT):
            self.__guess = lambda: self.mc_guess_inv_near_const(
                kwargs["consts"], kwargs["range"], kwargs["prev_I"], kwargs["change_size_prob"], kwargs["change_value_prob_ratio"])
        elif (strategy == GuessStrategy.MC_OCTAGONAL_DOMAIN):
            self.__guess = lambda: self.mc_guess_inv_oct(
                kwargs["consts"], kwargs["prev_I"], kwargs["change_size_prob"], kwargs["change_value_prob_ratio"])
        elif (strategy == GuessStrategy.MC_OCTAGONAL_DOMAIN_EXTENDED):
            self.__guess = lambda: self.mc_guess_inv_oct_ext(
                kwargs["consts"], kwargs["prev_I"], kwargs["change_size_prob"], kwargs["change_value_prob_ratio"])
        else:
            raise Exception(
                "Guess.__init__: unexpected guess strategy &s." % (strategy))

    def guess(self):
        return self.__guess()

    def guess_pred(self, coeff_dom, const_dom):
        def init(i):
            if i < self.num_var:
                return np.random.choice(coeff_dom)
            elif i == self.num_var:
                return np.random.choice(conf.OP_DOMAIN, p=self.op_dist)
            else:
                return np.random.choice(const_dom)
        # It doesn't matter if all coefficients of P are zero, then the predicate represents either True or False depending on the constant and operator value.
        return np.fromfunction(np.vectorize(init), (self.len_pred,), dtype=int)

    def guess_pred_oct(self, coeff_dom, const_dom):
        nz_pos = np.random.choice(
            self.num_var, self.NUM_OCT_NONZERO_POS, replace=False)

        def init(i):
            if i < self.num_var:
                if i in nz_pos:
                    return np.random.choice(coeff_dom)
                else:
                    return 0
            elif i == self.num_var:
                return np.random.choice(conf.OP_DOMAIN, p=self.op_dist)
            else:
                return np.random.choice(const_dom)
        # It doesn't matter if all coefficients of P are zero, then the predicate represents either True or False depending on the constant and operator value.
        return np.fromfunction(np.vectorize(init), (self.len_pred,), dtype=int)

    def guess_conj(self, coeff_dom, const_dom, num_conj, max_num_conj, is_oct_pred):
        assert(num_conj > 0)
        result = np.concatenate([self.guess_pred_oct(coeff_dom, const_dom)[np.newaxis] for _ in range(num_conj)]) if (is_oct_pred) else \
            np.concatenate([self.guess_pred(coeff_dom, const_dom)[
                           np.newaxis] for _ in range(num_conj)])
        return np.concatenate((result, np.zeros((max_num_conj-num_conj, self.num_var+2), dtype=int)))
    # TODO: assume we dont need the same conj numbers for now
    # def size_norm_conj(self, conjunctive_clause, max_num_conj):
    #     if (conjunctive_clause.shape[0] == max_num_conj):
    #         return conjunctive_clause
    #     return self.size_norm_conj(np.append(conjunctive_clause, np.array([0, 0, 0, -1, 0], ndmin=2), axis=0), max_num_conj)

    def guess_inv(self, coeff_dom, const_dom, is_oct_pred: bool,
                  max_num_disj=None, max_num_conj=None):
        if (max_num_disj is None):
            max_num_disj = self.max_num_disj
        if (max_num_conj is None):
            max_num_conj = self.max_num_conj
        num_disj = np.random.randint(1, max_num_disj + 1)
        if (num_disj == 0):
            return np.empty(shape=(0, max_num_conj, self.num_var+2), dtype=int)
        return np.concatenate([self.guess_conj(coeff_dom, const_dom, np.random.randint(1, max_num_conj + 1), max_num_conj, is_oct_pred)[np.newaxis] for _ in range(num_disj)])

    def guess_inv_small_const(self, max_const):
        const_dom = range(-max_const, max_const + 1)
        return self.guess_inv(const_dom, const_dom, False)

    def guess_inv_oct(self, consts: List[int]):
        consts.sort()
        return self.guess_inv([-1, 0, 1], consts, True)

    def guess_inv_oct_ext(self, consts: List[int]):
        consts.sort()
        return self.guess_inv([-1, 0, 1], consts, False)

    def guess_inv_near_const(self, values_in_program, k):
        val_set = set()
        for value in values_in_program:
            val_set |= set(range(value - k, value + k + 1))
        val_list = list(val_set)
        val_list.sort()
        return self.guess_inv(val_list, val_list, False)

    @staticmethod
    def get_geo_prob_list(domainList, centre, r):
        # This function assumes that domainList is sorted, and elements cannot be repeated
        def norm_neg_error_in_list(approx_list, neg_err):
            # Assumes negative_error is a negative value
            l = approx_list.copy()
            i = 0
            while i < len(l):
                if (neg_err >= 0):
                    break
                l[i] = max(0, l[i] + neg_err)
                neg_err += approx_list[i]
                i += 1
            else:
                if(conf.DISPLAY_WARNINGS == conf.ON):
                    print("Too much negative error in generating this list!")
                l.append(neg_err)
            return l

        def get_GP_list(a, r, N):
            ret = [a]
            for i in range(N-1):
                ret.append(ret[-1]*r)
            return ret

        index = domainList.index(centre)
        s1 = index
        s2 = len(domainList) - index - 1

        sublist1 = []
        sublist2 = []
        a = 1.0
        if (s1 == 0):
            a = ((float)(1 - r)) / (1 - r**s2)
            sublist1 = []
            sublist2 = get_GP_list(a, r, s2)
        elif (s2 == 0):
            a = ((float)(1 - r)) / (1 - r**s1)
            sublist1 = get_GP_list(a, r, s1)[::-1]
            sublist2 = []
        else:
            a = ((float)(1 - r)) / ((1 - r**s1) + (1 - r**s2))
            sublist1 = get_GP_list(a, r, s1)[::-1]
            sublist2 = get_GP_list(a, r, s2)

        probList = sublist1 + [0] + sublist2

        computation_error = 1 - sum(probList)
        if(computation_error >= 0):
            if (index == 0):
                probList[index + 1] = probList[index + 1] \
                    + (computation_error)
            elif (index == len(probList) - 1):
                probList[index - 1] = probList[index - 1] \
                    + (computation_error)
            else:
                probList[index - 1] = probList[index - 1] \
                    + (computation_error * 0.5)
                probList[index + 1] = probList[index + 1] \
                    + (computation_error * 0.5)
        else:
            probList = norm_neg_error_in_list(probList[0:index], 0.5 * computation_error) + probList[index: index + 1] + (
                norm_neg_error_in_list((probList[index + 1:])[::-1], 0.5 * computation_error))[::-1]
        return probList

    # Assumes prev_I has correct numpy dimensions.

    def mc_guess_inv(self, coeff_dom, const_dom, is_oct_pred,
                     prev_I, change_size_prob, change_value_prob_ratio):
        # Need sorted lists
        coeff_dom.sort()
        const_dom.sort()

        prev_num_disj = prev_I.shape[0]

        if np.random.rand() <= change_size_prob:
            domain_of_disjunct_values = list(range(1, self.max_num_disj + 1))
            domain_of_disjunct_values.remove(prev_num_disj)
            if not domain_of_disjunct_values:
                if (self.DISPLAY_WARNINGS == self.ON):
                    print(
                        "Size change not possible becuase domain of disjunct values is empty,")
                return prev_I
            new_num_disj = np.random.choice(
                domain_of_disjunct_values)
            if (new_num_disj < prev_num_disj):
                return prev_I[0:new_num_disj]
            else:
                additional_disjuncts = new_num_disj - prev_num_disj
                max_num_conj = prev_I.shape[1]
                return np.concatenate((prev_I, self.guess_inv(coeff_dom, const_dom, is_oct_pred, additional_disjuncts, max_num_conj)))
        else:  # size is constant:
            prev_CC_position = np.random.choice(
                range(0, prev_num_disj))
            prev_CC = prev_I[prev_CC_position]
            prev_CC_size = prev_CC.shape[0]
            prev_P_position = np.random.choice(range(0, prev_CC_size))
            prev_P = prev_CC[prev_P_position]
            index_to_change = np.random.choice(range(self.num_var+2))
            new_P = prev_P
            if (is_oct_pred == 0):
                if (index_to_change < self.num_var):
                    domain = coeff_dom
                elif (index_to_change == self.num_var):
                    domain = conf.OP_DOMAIN
                else:
                    domain = const_dom
                prob_list = Guess.get_geo_prob_list(
                    domain, prev_P[index_to_change], change_value_prob_ratio)
                newval = np.random.choice(domain, p=prob_list)
                new_P[index_to_change] = newval
            else:
                if (index_to_change < self.num_var):
                    domain = coeff_dom
                    prob_list = Guess.get_geo_prob_list(
                        domain, prev_P[index_to_change], change_value_prob_ratio)
                    newval = np.random.choice(domain, p=prob_list)
                    new_P[index_to_change] = newval
                    nonzero_index = (np.transpose(
                        np.nonzero(prev_P[0:self.num_var])).reshape(-1)).tolist()
                    if (len(nonzero_index) == 2 and index_to_change not in nonzero_index):
                        index_to_reset = np.random.choice(nonzero_index)
                        new_P[index_to_reset] = 0
                else:
                    if (index_to_change == self.num_var):
                        domain = conf.OP_DOMAIN
                    else:
                        domain = const_dom
                    prob_list = Guess.get_geo_prob_list(
                        domain, prev_P[index_to_change], change_value_prob_ratio)
                    newval = np.random.choice(domain, p=prob_list)
                    new_P[index_to_change] = newval
            new_CC = prev_CC
            new_CC[prev_P_position] = new_P
            I = prev_I
            I[prev_CC_position] = new_CC
            return I

    def mc_guess_inv_small_const(self, max_const, prev_I, change_size_prob, change_value_prob_ratio):
        const_dom = list(range(-max_const, max_const + 1))
        const_dom.sort()
        return self.mc_guess_inv(const_dom, const_dom, False,
                                 prev_I, change_size_prob, change_value_prob_ratio)

    def mc_guess_inv_oct(self, consts: List[int], prev_I, change_size_prob, change_value_prob_ratio):
        consts.sort()
        return self.mc_guess_inv([-1, 0, 1], consts, True,
                                 prev_I, change_size_prob, change_value_prob_ratio)

    def mc_guess_inv_oct_ext(self, consts: List[int], prev_I, change_size_prob, change_value_prob_ratio):
        consts.sort()
        return self.mc_guess_inv([-1, 0, 1], consts, True,
                                 prev_I, change_size_prob, change_value_prob_ratio)

    def mc_guess_inv_near_const(self, values_in_program, k, prev_I, change_size_prob, change_value_prob_ratio):
        val_set = set()
        for value in values_in_program:
            val_set |= set(range(value - k, value + k + 1))
        val_list = list(val_set)
        val_list.sort()
        return self.mc_guess_inv(val_list, val_list, False,
                                 prev_I, change_size_prob, change_value_prob_ratio)

    # print(random_guess_predicate([-1,1,0] , programConstants, np.array([0.2, 0.2, 0.2, 0.2, 0.2])) )
    # print(random_guess_octagonalpredicate([-1,1,0] , programConstants, np.array([0.2, 0.2, 0.2, 0.2, 0.2])) )
    # print(random_guess_conjunctive_clause([-1,1,0] , programConstants, 3,  np.array([0.2, 0.2, 0.2, 0.2, 0.2]), 0) )
    # print(size_normalize_conjunctive_clause(random_guess_conjunctive_clause([-1,1,0] , programConstants, 3,  np.array([0.2, 0.2, 0.2, 0.2, 0.2]), 1), 5 ) )
    # print( random_guess_inv([-1,1,0] , programConstants, 3, 4, np.array([0.2, 0.2, 0.2, 0.2, 0.2]), 1) )
    # print( random_guess_invariant([-1,1,0] , programConstants, 3, 4, np.array([0.2, 0.2, 0.2, 0.2, 0.2]), 1) )
    # print(guess_invariant_smallConstants(10, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) ))
    # print(guess_invariant_octagonaldomain(programConstants, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) ))
    # print(guess_invariant_octagonaldomain_extended(programConstants, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) ))
    # print(guess_invariant_nearProgramConstants(programConstants, 2, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) ))

    # Testing the MC functions:
    # I_p = guess_invariant_octagonaldomain(programConstants, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) )
    # print(I_p)
    # print(mc_guess_invariant(I_p, [-1,0,1], programConstants, 3, 0.1, 0.5, 1, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) ))

    # I_1 = guess_invariant_smallConstants(10, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) )
    # print(I_1)
    # print(mc_guess_invariant_smallConstants (I_1, 10, 3, 0.1, 0.5, np.array([0.2, 0.2, 0.2, 0.2, 0.2])))

    # I_2 = guess_invariant_octagonaldomain(programConstants, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) )
    # print(I_2)
    # print(mc_guess_invariant_octagonaldomain (I_2, programConstants, 3, 0.1, 0.5, np.array([0.2, 0.2, 0.2, 0.2, 0.2])))

    # I_3 = guess_invariant_octagonaldomain_extended(programConstants, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) )
    # print(I_3)
    # print(mc_guess_invariant_octagonaldomain_extended (I_3, programConstants, 3, 0.1, 0.5, np.array([0.2, 0.2, 0.2, 0.2, 0.2])))

    # I_4 = guess_invariant_nearProgramConstants(programConstants, 5, 3, 3, np.array([0.2, 0.2, 0.2, 0.2, 0.2]) )
    # print(I_4)
    # print(mc_guess_invariant_nearProgramConstants(I_4, programConstants, 5, 3, 0.1, 0.5, np.array([0.2, 0.2, 0.2, 0.2, 0.2])))

# a = Guess(1, 1, 1, [],1)
# x = a.get_geo_prob_list([i for i in range(5)], 4, 0.5)
# print(x, 1 - sum(x))

''' ********************************************************************************************************************'''


def op_norm_pred(P):
    n = len(P) - 2
    if (P[n] > 0):
        return np.array(np.multiply(P, -1), ndmin=2)
    elif (P[n] in [0, 10, -10] ):
        temp1, temp2 = np.multiply(P, -1), P
        if (P[n] == 0):
            temp1[n] = -1
            temp2[n] = -1
        else:
            temp1[n] = -2
            temp2[n] = -2            
        return np.array([temp1, temp2], ndmin=2)
    return np.array(P, ndmin=2)


def op_norm_conj(C):
    assert(len(C) > 0)  # assuming C not empty
    return np.concatenate([op_norm_pred(C[i]) for i in range(len(C))])


def norm_disj(D, conjunct_size):
    n = len(D) - 2
    if (n == 0):
        return np.empty(shape=(0, conjunct_size, n+2), dtype=int)
    C = op_norm_conj(D[0])
    
    padding_pred = np.zeros(n + 2)
    padding_pred[n] = -1
    while (len(C) <= conjunct_size):
        C = np.concatenate((C, np.array(padding_pred, ndmin=2)))
    return np.concatenate((np.array(C, ndmin=3), norm_disj(D[1:], conjunct_size)))


def norm_DNF(D, n):
    """
    :n: The number of variables.
    """
    # Let operator_normalized versions only have <= or < operators.

    max_size = 0
    for C in D:
        curr_size = 0
        for P in C:
            curr_size = curr_size + (2 if (P[n] == 0) else 1)
        max_size = max(max_size, curr_size)
    return norm_disj(D, max_size)

''' ********************************************************************************************************************'''

    """ Cost functions.
    """

    def __J1(self, cex_list):
        error = 0
        for cex in cex_list:
            # TODO: x%i is assumed across modules, should make it globally configured.
            pt = [cex.evaluate(Int("x%i"), model_completion=True).as_long()
                  for i in range(self.num_var)]
            point = np.array(pt)
            error = max(error, self.__distance_point_DNF(point, self.I))
        return error + len(cex_list)

    # Traditionally try to 'guess' which cex are supposed to be negative, and which are supposed to be positive, and then there is a relative ratio; but we skip that here.

    def __J2(self, cex_list):
        return len(cex_list)

    def __J3(self, cex_list):
        error = 0
        for cex in cex_list:
            pt = [cex.evaluate(Int("x%i"), model_completion=True).as_long()
                  for i in range(self.num_var)]
            point = np.array(pt)
            error = max(error, self.__distance_point_DNF(point, self.Q))
        return error + len(cex_list)

    """ Distance Functions.
    """

    def __distance_point_DNF(self, p, D: np.ndarray):
        """ 
        :point: an 1d array with len = num_var.
        :D: a DNF as np.ndarray.
        """
        assert(len(p) == self.num_var)

        def distance_point_conj_clauses(C):
            """
            :C: a 2d array, (conj, pred).
            """
            C = op_norm_conj(C)
            A = np.concatenate(
                [C[:, :self.num_var], C[:, self.num_var+1:]], axis=1)
            return float(minimize(
                lambda x, p: np.linalg.norm(x - p),
                np.zeros(self.num_var),
                args=(p,),
                constraints=[LinearConstraint(A[:, :-1], -np.inf, -A[:, -1])],
            ).fun)

        d = float('inf')
        for C in D:
            d = min(d, distance_point_conj_clauses(C))
        return d

    # Testing:
    # print(distance_point_conj_clauses( np.array( [-3,1,3], ndmin = 1), np.array( [ [1,2,3,1,10], [1,3,1,0,0] ] , ndmin = 2)) )
    # print(distance_point_DNF( np.array( [-3,3,1], ndmin = 1), np.array( [ [ [-7,1,3,-2,3], [1,2,1,0,2], [3,1,3, 1, 4] ], [ [1,3,1,1,10], [1,3,1,0,0], [0,0,0,0,0] ] ]    )) )


''' ********************************************************************************************************************'''

class Cost:
    def __init__(self, repr: Repr, I_arr: np.ndarray):
        self.num_var = repr.get_num_var()
        self.num_cex = conf.NUM_COUNTEREXAMPLE
        self.I = I_arr
        self.I_z3 = DNF_to_z3expr(I_arr)
        self.Ip_z3 = DNF_to_z3expr_p(I_arr)
        self.P = repr.get_P()
        self.B = repr.get_B()
        self.T = repr.get_T()
        self.Q = repr.get_Q()
        self.P_z3 = repr.get_P_z3expr()
        self.B_z3 = repr.get_B_z3expr()
        self.T_z3 = repr.get_T_z3expr()
        self.Q_z3 = repr.get_Q_z3expr()

        C1_cex_list = self.__get_cex_C1()
        C2_cex_list = self.__get_cex_C2()
        C3_cex_list = self.__get_cex_C3()

        # Get costFunction
        cost1 = self.__J1(C1_cex_list)
        cost2 = self.__J2(C2_cex_list)
        cost3 = self.__J3(C3_cex_list)
        self.cost = conf.K1*cost1 + conf.K2*cost2 + conf.K3*cost3

    def get_cost(self):
        return self.cost

    """ Get counterexamples.
    """

''' ********************************************************************************************************************'''

def guess_inv(repr: Repr, max_guesses, guess_strat, max_const=None, guess_range=None):
    cost = float('inf')
    count_guess = 0
    num_var = repr.get_num_var()
    I = None
    while (cost != 0 and count_guess < max_guesses):
        count_guess += 1
        guesser = Guess(num_var, parameters.max_conjuncts, parameters.max_disjuncts,
                        np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                        guess_strat,
                        max_const=max_const,
                        range=guess_range,
                        consts=repr.get_consts())
        I = guesser.guess()
        cost = Cost(repr, I).get_cost()
        if (parameters.PRINT_ITERATIONS == parameters.ON):
            print(count_guess, '   ', DNF_to_z3expr(I), "\t", end='')
            print('   ', round(cost, 2))
    if (parameters.PRINT_ITERATIONS != parameters.ON):
        print(count_guess, '   ', DNF_to_z3expr(I), "\t", end='')
        print('   ', round(cost, 2))
    return (I, cost, count_guess)





def mc_guess_inv(repr: Repr, max_guesses, guess_strat, max_const=None, guess_range=None, change_size_prob=None, change_value_prob_ratio=None):
    count_guess = 0
    cost = float('inf')
    num_var = repr.get_num_var()
    I = None
    while (cost > 0 and count_guess < max_guesses):
        count_guess += 1

        if (count_guess == 1):
            (I, cost, count_guess) = guess_inv(repr, 1, GuessStrategy.mc_to_not_mc(
                guess_strat), max_const=max_const, guess_range=guess_range)
            continue

        prev_cost = cost
        prev_I = I
        guesser = Guess(num_var, parameters.max_conjuncts, parameters.max_disjuncts,
                        np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                        guess_strat,
                        prev_I=prev_I,
                        max_const=max_const,
                        range=guess_range,
                        consts=repr.get_consts(),
                        change_size_prob=change_size_prob,
                        change_value_prob_ratio=change_value_prob_ratio)
        I = guesser.guess()  # I
        cost = Cost(repr, I).get_cost()

        if (cost >= prev_cost):
            # 0.05 is subtracted so that prob_of_staying is not zero if curr_cost = prev_cost
            change_prob = max((prev_cost/cost) - 0.05, 0.0)
            # Don't change case
            if(np.random.rand() > change_prob):
                cost = prev_cost
                I = prev_I
            else:
                if (parameters.PRINT_ITERATIONS == parameters.ON):
                    print(count_guess, '   ', DNF_to_z3expr(I), "\t", end='')
                    print('   ', round(cost, 2))
        else:
            if (parameters.PRINT_ITERATIONS == parameters.ON):
                print(count_guess, '   ', DNF_to_z3expr(I), "\t", end='')
                print('   ', round(cost, 2))
    if (parameters.PRINT_ITERATIONS != parameters.ON):
        print(count_guess, '   ', DNF_to_z3expr(I), "\t", end='')
        print('   ', round(cost, 2))
    return (I, cost, count_guess)


def run_all_strategies(program, iterations, run_random_strategies, run_MCMC_strategies, max_const, small_guess_range, large_guess_range, change_size_prob, change_value_prob_ratio):
    if (run_random_strategies):
        random_iterations = {}
        random_strategies = {"R_Small_Constant": GuessStrategy.SMALL_CONSTANT, "R_Octagonal_Domain": GuessStrategy.OCTAGONAL_DOMAIN,
                             "R_Octagonal_Domain_Extended":  GuessStrategy.OCTAGONAL_DOMAIN_EXTENDED, "R_NearConstant_small": GuessStrategy.NEAR_CONSTANT,
                             "R_NearConstant_Large": GuessStrategy.NEAR_CONSTANT}
        for strategy in random_strategies:
            sum = 0.0
            timeouts = 0
            for _ in range(iterations):
                (_, cost, val) = guess_inv(program, parameters.max_guesses, random_strategies[strategy], max_const=max_const,
                                           guess_range=(small_guess_range if strategy == "R_NearConstant_small" else
                                                        (large_guess_range if strategy == "R_NearConstant_Large" else None)))
                sum = sum + (1.0 * val)
                if (cost != 0):
                    timeouts = timeouts + 1
            random_iterations[strategy] = (sum/iterations, timeouts)

        print("\n\nRANDOM_GUESSES:", random_iterations.keys(),
              '\n', random_iterations.values())

    if(run_MCMC_strategies):

        MCMC_iterations = {}
        MCMC_strategies = {"MCMC_Small_Constant": GuessStrategy.MC_SMALL_CONSTANT, "MCMC_Octagonal_Domain": GuessStrategy.MC_OCTAGONAL_DOMAIN,
                           "MCMC_Octagonal_Domain_Extended":  GuessStrategy.MC_OCTAGONAL_DOMAIN_EXTENDED, "MCMC_NearConstant_small": GuessStrategy.MC_NEAR_CONSTANT,
                           "MCMC_NearConstant_Large": GuessStrategy.MC_NEAR_CONSTANT}
        for strategy in MCMC_strategies:
            sum = 0.0
            timeouts = 0
            for _ in range(iterations):
                (_, cost, val) = mc_guess_inv(program, parameters.max_guesses, MCMC_strategies[strategy], max_const=max_const,
                                              guess_range=(small_guess_range if strategy == "MCMC_NearConstant_small" else
                                                           (large_guess_range if strategy == "MCMC_NearConstant_Large" else None)),
                                              change_size_prob=change_size_prob, change_value_prob_ratio=change_value_prob_ratio)
                sum = sum + (1.0 * val)
                if (cost != 0):
                    timeouts = timeouts + 1
            MCMC_iterations[strategy] = (sum/iterations, timeouts)
        print("MCMC_GUESSES:", MCMC_iterations.keys(),
              '\n', MCMC_iterations.values())


program = input.mock.mock4


# guess_inv(repr = program, max_guesses = 1, guess_strat = GuessStrategy.SMALL_CONSTANT, max_const=5, guess_range=None)

run_all_strategies(program, iterations=1, run_random_strategies=1, run_MCMC_strategies=0, max_const=10,
                   small_guess_range=1, large_guess_range=5, change_size_prob=0.1, change_value_prob_ratio=0.5)



''' ********************************************************************************************************************'''

                   # Returns a list of list of (n+1)-element-lists.
def deg_list(I, Dp):
    n = len(I[0][0]) - 2
    def degcc_list(cc, Dp, n):
        def degp_list(p, Dp, n):
            ret = []
            for i in range(n+2):
                if (i < n):
                    if (p[i] == min(Dp[0]) or p[i] == max(Dp[0]) ):
                        ret.append(1)
                    else:
                        ret.append(2)
                elif (i == n):
                    continue
                else:
                    if (p[i] == min(Dp[1]) or p[i] == max(Dp[1]) ):
                        ret.append(1)
                    else:
                        ret.append(2)
            return ret
        return [degp_list(p, Dp, n) for p in cc ]
    return [degcc_list(cc, Dp, n) for cc in I ]

def deg(deglist):
    return sum([ sum([ sum(p) for p in cc  ]) for cc in deglist])


def uniformlysampleLII(Dp, c, d, n, samplepoints, beta):
    def uniformlysampleLIcc(Dp, n, c):
        def uniformlysampleLIp(Dp, n):
            def uniformlysamplenumber(i):
                if i < n:
                    return np.random.choice(Dp[0])
                elif i == n:
                    return -1
                else:
                    return np.random.choice(Dp[1])
            return np.fromfunction(np.vectorize(uniformlysamplenumber), shape = (n+2,), dtype=int)
        
        cc = np.empty(shape=(0,n+2), dtype = 'int')
        for i in range(c):
            cc = np.concatenate((cc, np.array([uniformlysampleLIp(Dp,n)], ndmin=2)))
        return cc

    I = [uniformlysampleLIcc(Dp, n, c) for i in range(d) ]
    (fI, costI, costtuple) = f(I, samplepoints, beta )
    return (I, deg_list(I, Dp), fI, costI, costtuple)


def randomwalktransition(I_prev, deglist_I, Dp, samplepoints, costtuple_I, beta):
    # i is a number from 1 to degree
    def ithneighbor(I_old, i, deglist, Dp):
        I = I_old.copy()
        k = i
        n = len(I[0][0]) - 2
        for ccindex, degcc_list in enumerate(deglist):
            for pindex, degp_list in enumerate(degcc_list):
                if ( k > sum(degp_list)):
                    k = k - sum(degp_list)
                    continue
                else:
                    index = (ccindex, pindex)
                    for vindex, deg in enumerate(degp_list):
                        if (k > deg):
                            k = k - deg
                            continue
                        else:
                            vindex_actual = vindex if (vindex < n) else (vindex + 1)                  
                            if (deg == 2):
                                I[ccindex][pindex][vindex_actual] = I[ccindex][pindex][vindex_actual] + (1 if (k == 1) else -1)
                            else: 
                                j = 0 if (vindex_actual < n+1) else 1
                                r = 1 if (min(Dp[j]) == I[ccindex][pindex][vindex_actual]) else -1
                                I[ccindex][pindex][vindex_actual] = I[ccindex][pindex][vindex_actual] + r
                            return (I, index)
    I = deepcopy_DNF(I_prev) #deepcopy here!
    degree = deg(deglist_I) 
    i = np.random.choice(range(1, degree+1,1))
    (Inew, index) = ithneighbor(I, i, deglist_I, Dp)
    (fnew, costnew, costtuplenew) = f(Inew, samplepoints, beta, costtuple_I, index)
    return (Inew, copy.deepcopy(deg_list(Inew, Dp)), fnew, costnew, costtuplenew)


# Testing
# plus = [ [0] ]
# minus = [ [7], [10000] ]
# ICE = [ ( [5] , [6]  )  ]
# samplepoints = (plus, minus, ICE)
# (I, deglistI, fI, costI, costtupleI) = uniformlysampleLII( (list(range(-10, 10, 1)), list(range(-10,10,1)) ), 1, 1, 1, samplepoints )
# print(I, deglistI, fI, costI, costtupleI) 
# Dp = (range(-10, 10, 1), range(-10,10,1) )
# print(randomwalktransition(I, deglistI, Dp, samplepoints, costtupleI ))







''' ********************************************************************************************************************'''





class setType:
    plus = "plus"
    minus = "minus"
    ICE = "ICE"




def LIPptdistance(p, pt):
    return max( sum(p[:-2]* pt) - p[-1] , 0)



def d(p, pt, pt_type):
    if (pt_type == setType.plus):
        return LIPptdistance(p, pt)
    elif (pt_type == setType.minus):
        return LIPptdistance(negationLIpredicate(p), pt)
    else:
        return min( LIPptdistance(negationLIpredicate(p), pt[0]) , LIPptdistance(p, pt[1])  )

def U(r, U_type):
    if (U_type == setType.plus):
        return 1.0
    elif (U_type == setType.minus):
        return 1.0
    else:
        return 1.0 

def cost_sum(costlist):
    return sum ([ min([ sum([p for p in cc]) for cc in pt_I]) for pt_I in costlist ])

def cost_max(costlist):
    return max ([ min([ sum([p for p in cc]) for cc in pt_I]) for pt_I in costlist ])

def costtuple(I, S, set_type ):
    if (set_type == setType.plus):
        costlist = [ [ [LIPptdistance(p, pt) for p in cc  ] for cc in I ]  for pt in S]
        return (cost_sum(costlist), costlist)
    elif (set_type == setType.minus):
        costlist = [ [ [LIPptdistance(negationLIpredicate(p), pt) for p in cc  ] for cc in I ]  for pt in S] 
        return (cost_sum(costlist), costlist)
    else:
        costlist = ([ [ [LIPptdistance(negationLIpredicate(p), pt[0]) for p in cc  ] for cc in I ]  for pt in S], 
                            [ [ [LIPptdistance(p, pt[1]) for p in cc  ] for cc in I ]  for pt in S] )
        return ( min( cost_sum(costlist[0]), cost_sum(costlist[1]) ), costlist)    


def optimized_costtuple(I, S, set_type, prev_costlist, inv_i):
    costlist = deepcopy(prev_costlist)
    if (set_type == setType.plus or set_type == setType.minus):
        for j,pt in enumerate(S):
            costlist[j][inv_i[0]][inv_i[1]] = d(I[inv_i[0]][inv_i[1]], pt, set_type)
        return (cost_sum(costlist), costlist)
    else:
        for j,pt in enumerate(S):
            costlist[0][j][inv_i[0]][inv_i[1]] = d(I[inv_i[0]][inv_i[1]], pt, set_type)        
            costlist[1][j][inv_i[0]][inv_i[1]] = d(I[inv_i[0]][inv_i[1]], pt, set_type) 
        return ( min( cost_sum(costlist[0]), cost_sum(costlist[1]) ), costlist) 