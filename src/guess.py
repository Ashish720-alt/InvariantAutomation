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
                return np.append(prev_I, self.guess_inv(coeff_dom, const_dom, is_oct_pred, max_num_conj,
                                                        additional_disjuncts), axis=0)
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
