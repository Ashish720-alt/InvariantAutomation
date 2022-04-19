""" Imports.
"""
from input import get_input
from configure import Configure as conf
from cost_funcs import Cost
from dnf import DNF_to_z3expr, DNF_to_z3expr_p
from guess import Guess, GuessStrategy
from repr import Repr
from z3 import *
import numpy as np


""" Main function.
"""


def guess_inv(repr: Repr, max_guesses, guess_strat, max_const=None, guess_range=None):
    cost = float('inf')
    count_guess = 0
    num_var = repr.get_num_var()
    I = None
    while (cost != 0 and count_guess < max_guesses):
        count_guess += 1
        guesser = Guess(num_var, conf.max_conjuncts, conf.max_disjuncts,
                        np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                        guess_strat,
                        max_const=max_const,
                        range=guess_range,
                        consts=repr.get_consts())
        I = guesser.guess()
        cost = Cost(repr, I).get_cost()
        print(count_guess, '   ', DNF_to_z3expr(I), "\t", end='')
        print('   ', round(cost, 2))
    return (I, cost)



def mc_guess_inv(repr: Repr, max_guesses, guess_strat, max_const=None, guess_range=None, change_size_prob=None, change_value_prob_ratio=None):
    count_guess = 0
    cost = float('inf')
    num_var = repr.get_num_var()
    I = None
    while (cost > 0 and count_guess < max_guesses):
        count_guess += 1

        if (count_guess == 1):
            (I, cost) = guess_inv(repr, 1, GuessStrategy.mc_to_not_mc(guess_strat), max_const=max_const, guess_range=guess_range)
            continue

        prev_cost = cost
        prev_I = I
        guesser = Guess(num_var, conf.max_conjuncts, conf.max_disjuncts,
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
                print(count_guess, '   ', DNF_to_z3expr(I), "\t", end='')
                print('   ', round(cost, 2))
        else:
            print(count_guess, '   ', DNF_to_z3expr(I), "\t", end='')
            print('   ', round(cost, 2))            
    return



guess_inv(
    get_input(P=np.array([[[1, 0, 0]]]),
              B=np.array([[[1, -2, 6]]]),
              Q=np.array([[[1, 0, 6]]]),
              T=np.array([[1, 1], [0, 1]])),
    conf.max_guesses,
    GuessStrategy.SMALL_CONSTANT,
     max_const=10
)




# guess_inv(
#     get_input(P=np.array([[[1, 0, 0]]]),
#               B=np.array([[[1, -2, 6]]]),
#               Q=np.array([[[1, 0, 6]]]),
#               T=np.array([[1, 1], [0, 1]])),
#     conf.max_guesses,
#     GuessStrategy.OCTAGONAL_DOMAIN
# )

# guess_inv(
#     get_input(P=np.array([[[1, 0, 0]]]),
#               B=np.array([[[1, -2, 6]]]),
#               Q=np.array([[[1, 0, 6]]]),
#               T=np.array([[1, 1], [0, 1]])),
#     conf.max_guesses,
#     GuessStrategy.OCTAGONAL_DOMAIN_EXTENDED
# )

# guess_inv(
#     get_input(P=np.array([[[1, 0, 0]]]),
#               B=np.array([[[1, -2, 6]]]),
#               Q=np.array([[[1, 0, 6]]]),
#               T=np.array([[1, 1], [0, 1]])),
#     conf.max_guesses,
#     GuessStrategy.NEAR_CONSTANT,
#     guess_range=1
# )

# mc_guess_inv(
#     get_input(P=np.array([[[1, 0, 0, 0, 0]]]),
#               B=np.array([[[1, 0, 0, -2, 6]]]),
#               Q=np.array([[[1, 0, 0, 0, 6]]]),
#               T=np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])),
#     conf.max_guesses,
#     GuessStrategy.MC_NEAR_CONSTANT,
#     guess_range=1,
#     change_size_prob=0.1,
#     change_value_prob_ratio=0.5
# )
