""" Guessing a new invriant.
"""
import numpy as np
from configure import Configure as conf
from typing import List
from enum import Enum, auto


class GuessStrategy(Enum):
    SMALL_CONSTANT = auto()
    NEAR_CONSTANT = auto()
    OCTAGONAL_DOMAIN = auto()
    OCTAGONAL_DOMAIN_EXTENDED = auto()

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

        self.NUM_OCT_NONZERO_POS = 2
        
        self.__guess = None
        if (strategy == GuessStrategy.SMALL_CONSTANT):
            self.__guess = lambda: self.guess_inv_small_const(kwargs["max_const"])
        elif (strategy == GuessStrategy.NEAR_CONSTANT):
            self.__guess = lambda: self.guess_inv_near_const(kwargs["consts"], kwargs["range"])
        elif (strategy == GuessStrategy.OCTAGONAL_DOMAIN):
            self.__guess = lambda: self.guess_inv_oct(kwargs["consts"])
        elif (strategy == GuessStrategy.OCTAGONAL_DOMAIN_EXTENDED):
            self.__guess = lambda: self.guess_inv_oct_ext(kwargs["consts"])

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

    def guess_conj(self, coeff_dom, const_dom, num_conj, is_oct_pred):
        assert(num_conj > 0)
        if (is_oct_pred):
            return np.concatenate([self.guess_pred_oct(coeff_dom, const_dom)[np.newaxis] for _ in range(num_conj)])
        else:
            return np.concatenate([self.guess_pred(coeff_dom, const_dom)[np.newaxis] for _ in range(num_conj)])

    # TODO: assume we dont need the same conj numbers for now
    # def size_norm_conj(self, conjunctive_clause, max_num_conj):
    #     if (conjunctive_clause.shape[0] == max_num_conj):
    #         return conjunctive_clause
    #     return self.size_norm_conj(np.append(conjunctive_clause, np.array([0, 0, 0, -1, 0], ndmin=2), axis=0), max_num_conj)

    def guess_DNF(self, coeff_dom, const_dom, is_oct_pred: bool,
                  max_num_disj=None, max_num_conj=None):
        if (max_num_disj is None):
            max_num_disj = self.max_num_disj
        if (max_num_conj is None):
            max_num_conj = self.max_num_conj
        num_disj = np.random.randint(1, max_num_disj + 1)
        if (num_disj == 0):
            return np.empty(shape=(0, max_num_conj, self.num_var+2), dtype=int)
        num_conj = np.random.randint(1, max_num_conj + 1)
        return np.concatenate([self.guess_conj(coeff_dom, const_dom, num_conj, is_oct_pred)[np.newaxis] for _ in range(num_disj)])

    def guess_inv_small_const(self, max_const):
        const_dom = range(-max_const, max_const + 1)
        return self.guess_DNF(const_dom, const_dom, False)

    def guess_inv_oct(self, consts: List[int]):
        return self.guess_DNF([-1, 0, 1], consts, True)

    def guess_inv_oct_ext(self, consts: List[int]):
        return self.guess_DNF([-1, 0, 1], consts, False)

    def guess_inv_near_const(self, values_in_program, k):
        val_set = set()
        for value in values_in_program:
            val_set |= set(range(value - k, value + k + 1))
        val_list = list(val_set)
        return self.guess_DNF(val_list, val_list, False)
        
    def get_GP_list(self, a, r, N):
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

    def __norm_neg_error_in_list(self, approximated_list, negative_error):
        if (negative_error >= 0):
            return approximated_list
        elif (approximated_list == []):
            if(conf.DISPLAY_WARNINGS == conf.ON):
                print("Too much negative error in generating this list!")
            return [negative_error]
        return [max(0, approximated_list[0] + negative_error)] + self.__norm_neg_error_in_list(approximated_list[1:], negative_error + approximated_list[0])

    # This function assumes that domainList is sorted, and elements cannot be repeated

    def __get_geo_prob_list(self, domainList, centre, r):
        index = domainList.index(centre)
        s1 = index
        s2 = len(domainList) - index - 1

        sublist1 = []
        sublist2 = []
        a = 1.0
        if (s1 == 0):
            a = ((float)(1 - r)) / (1 - r**s2)
            sublist1 = []
            sublist2 = self.__get_GP_list(a, r, s2)
        elif (s2 == 0):
            a = ((float)(1 - r)) / (1 - r**s1)
            sublist1 = self.__get_GP_list(a, r, s1)[::-1]
            sublist2 = self.__get_GP_list(a, r, s2)
        else:
            a = ((float)(1 - r)**2) / (1 - r**s1) * (1 - r**s2)
            sublist1 = self.__get_GP_list(a, r, s1)[::-1]
            sublist2 = self.__get_GP_list(a, r, s2)

        probList = sublist1 + [0] + sublist2

        computation_error = 1 - sum(probList)
        if(computation_error >= 0):
            if (index == 0):
                probList[index + 1] = probList[index +
                                               1] + (computation_error)
            elif (index == len(probList) - 1):
                probList[index - 1] = probList[index -
                                               1] + (computation_error)
            else:
                probList[index - 1] = probList[index -
                                               1] + (computation_error * 0.5)
                probList[index + 1] = probList[index +
                                               1] + (computation_error * 0.5)
        else:
            probList = self.__norm_neg_error_in_list(probList[0:index], 0.5 * computation_error) + probList[index: index + 1] + (
                self.__norm_neg_error_in_list((probList[index + 1:])[::-1], 0.5 * computation_error))[::-1]
        return probList

    # Implement the is_oct_pred version too!
    # Assumes I_prev has correct numpy dimensions.

    def MC_guess_inv(self, I_prev, coeff_dom, const_dom, max_num_disj,
                     change_size_prob, change_value_prob_GPratio, is_oct_pred):
        # Need sorted lists
        coeff_dom.sort()
        conf.OP_DOMAIN = [-2, -1, 0, 1, 2]
        const_dom.sort()

        prev_num_disj = I_prev.shape[0]

        if(np.random.choice([0, 1], p=np.array([1 - change_size_prob, change_size_prob])) == 1):
            domain_of_disjunct_values = (
                list((range(1, max_num_disj + 1))))
            domain_of_disjunct_values.remove(prev_num_disj)
            if not domain_of_disjunct_values:
                if (self.DISPLAY_WARNINGS == self.ON):
                    print(
                        "Size change not possible becuase domain of disjunct values is empty,")
                return I_prev
            new_num_disj = np.random.choice(
                domain_of_disjunct_values)
            if (new_num_disj < prev_num_disj):
                return I_prev[0:new_num_disj]
            else:
                additional_disjuncts = new_num_disj - prev_num_disj
                max_num_conj = I_prev.shape[1]
                return np.append(I_prev, self.guess_DNF(coeff_dom, const_dom, is_oct_pred, max_num_conj,
                                                        additional_disjuncts), axis=0)
        else:  # size is constant:
            prev_CC_position = np.random.choice(
                range(0, prev_num_disj))
            prev_CC = I_prev[prev_CC_position]
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
                prob_list = self.__get_geo_prob_list(
                    domain, prev_P[index_to_change], change_value_prob_GPratio)
                newval = np.random.choice(domain, p=prob_list)
                new_P[index_to_change] = newval
            else:
                if (index_to_change < self.num_var):
                    domain = coeff_dom
                    prob_list = self.__get_geo_prob_list(
                        domain, prev_P[index_to_change], change_value_prob_GPratio)
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
                    prob_list = self.__get_geo_prob_list(
                        domain, prev_P[index_to_change], change_value_prob_GPratio)
                    newval = np.random.choice(domain, p=prob_list)
                    new_P[index_to_change] = newval
            new_CC = prev_CC
            new_CC[prev_P_position] = new_P
            I = I_prev
            I[prev_CC_position] = new_CC
            return I


    def MC_guess_inv_small_const(self, I_prev, max_const, max_num_disj, change_size_prob, change_value_prob_GPratio):
        const_dom = list(
            range(-max_const, max_const + 1))
        return self.MC_guess_inv(I_prev, const_dom, const_dom, max_num_disj, change_size_prob,
                                 change_value_prob_GPratio, False)

    def MC_guess_inv_oct(self, I_prev, consts: List[int], max_num_disj, change_size_prob, change_value_prob_GPratio):
        return self.MC_guess_inv(I_prev, [-1, 0, 1], consts, max_num_disj, change_size_prob,
                                 change_value_prob_GPratio, True)

    def MC_guess_inv_oct_ext(self, I_prev, consts: List[int], max_num_disj, change_size_prob, change_value_prob_GPratio):
        return self.MC_guess_inv(I_prev, [-1, 0, 1], consts, max_num_disj, change_size_prob,
                                 change_value_prob_GPratio, False)

    def MC_guess_inv_near_const(self, I_prev, values_in_program, k, max_num_disj, change_size_prob, change_value_prob_GPratio):
        list_of_values = []
        for value in values_in_program:
            list_of_values = list_of_values + \
                list(range(value - k, value + k + 1))
        list_of_values = list(set(list_of_values))  # remove duplicates
        return self.MC_guess_inv(I_prev, list_of_values, list_of_values, max_num_disj, change_size_prob,
                                 change_value_prob_GPratio, False)

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
