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

def guess_inv(repr: Repr, guess_strat, max_const=None, guess_range=None):
    cost = float('inf')
    count_guess = 0
    num_var = repr.get_num_var()
    I_arr = None
    while (cost != 0 and count_guess < conf.max_guesses):
        count_guess += 1
        guesser = Guess(num_var, conf.max_conjuncts, conf.max_disjuncts,
                        np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                        guess_strat,
                        max_const=max_const,
                        range=guess_range,
                        consts=repr.get_consts())
        I_arr = guesser.guess()  # I
        cost = Cost(repr, I_arr).get_cost()
        # print(cost1, cost2, cost3, end = '')
        print('   ', round(cost, 2), '\n')
    return (I_arr, cost)


guess_inv(
    get_input(P=np.array([[[1, 0, 0, 0, 0]]]),
              B=np.array([[[1, 0, 0, -2, 6]]]),
              Q=np.array([[[1, 0, 0, 0, 6]]]),
              T=np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])),
    GuessStrategy.OCTAGONAL_DOMAIN
)


# MC_invariant_guess(max_guesses, (3, 1), max_conjuncts, max_disjuncts, 0.1, 0.5)


def MC_invariant_guess(guesses, guess_strat, num_conj, num_disj, k, r):
    count_guess = 0
    curr_cost = float('inf')
    curr_I = np.empty(shape=(0, max_conjuncts, n+2))
    while (curr_cost != 0 and count_guess < guesses):
        count_guess = count_guess + 1
        prev_cost = curr_cost
        prev_I = curr_I
        if (count_guess == 1):
            rv = random_invariant_guess(1, guess_strat, num_conj, num_disj)
            curr_cost = rv[1]
            curr_I = rv[0]
            continue
        else:
            if (guess_strat[0] == 1):
                curr_I = MC_guess_inv_smallConstants(
                    prev_I, guess_strat[1], num_disj, k, r, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
            elif (guess_strat[0] == 2):
                if (guess_strat[1] == 0):
                    curr_I = MC_guess_inv_octagonaldomain(
                        prev_I, programConstants, num_disj, k, r, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
                else:
                    curr_I = MC_guess_inv_octagonaldomain_extended(
                        prev_I, programConstants, num_disj, k, r, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
            elif (guess_strat[0] == 3):
                curr_I = MC_guess_inv_nearProgramConstants(
                    prev_I, programConstants, guess_strat[1], num_disj, k, r, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))

            curr_I_g = convert_DNF_to_lambda(curr_I)
            C1_cexList = GenerateCexList_C1(curr_I_g, s)
            # print(C1_cexList)
            C2_cexList = GenerateCexList_C2(curr_I_g, s)
            # print(C2_cexList)
            C3_cexList = GenerateCexList_C3(curr_I_g, s)
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
                # 0.05 is added so that prob_of_change is not zero if curr_cost = prev_cost
                prob_of_change = min(1.0 - (curr_cost/prev_cost) + 0.05, 1.0)
                # Don't change
                if(np.random.choice([0, 1], p=np.array([1 - prob_of_change, prob_of_change])) == 0):
                    curr_cost = prev_cost
                    curr_I = prev_I  # Isn't it pointer assignment here? Won't this give a really bad error, the same as the semantic bug as before?!?
                else:
                    print(count_guess, '   ', end='')
                    print_DNF(curr_I, 0)
                    print("\t", end='')
                    # print(cost1, cost2, cost3, end = '')
                    print('   ', round(curr_cost, 2), '\n')
    return
