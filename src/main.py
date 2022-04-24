""" Imports.
"""
from input import get_input
from configure import Configure as conf
from cost_funcs import Cost
from dnf import DNF_to_z3expr, DNF_to_z3expr_p
from guess import Guess, GuessStrategy
from repr import Repr
import repr
from z3 import *
import numpy as np
from math import floor

parameters = conf()

""" Main function. """


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


program = get_input(P=np.array([[[1, 0, 0, 0, 1], [0, 0, 1, 0, 0], [0, 1, 0, 1, 1], [0, 1, 0, -1, 1000]]]),
                    B=np.array([[[1, -1, 0, -2, 0]]]),
                    Q=np.array([[[0, -1, 1, 1, 0]]]),
                    # now we directly call it from repr, ideally we should do it in get_input(C_source_code)
                    T=repr.SimpleTotalTransitionFunc(np.array([[1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]])))


guess_inv(repr = program, max_guesses = 1, guess_strat = GuessStrategy.SMALL_CONSTANT, max_const=5, guess_range=None)

# run_all_strategies(program, iterations=1, run_random_strategies=0, run_MCMC_strategies=1, max_const=10,
#                    small_guess_range=1, large_guess_range=5, change_size_prob=0.1, change_value_prob_ratio=0.5)


