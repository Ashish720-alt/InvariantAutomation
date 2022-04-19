
""" Hyper Parameters.
"""
import operator

class Configure:
    s = 10

    
    # cost function related
    NUM_COUNTEREXAMPLE = 10 
    K1 = 1
    K2 = 2
    K3 = 1

    SB = 5
    SD = 3

    max_guesses = 1000

    max_disjuncts = 10
    max_conjuncts = 10

    # Print warnings or not?
    ON = 1
    OFF = 0
    DISPLAY_WARNINGS = ON

    # Operators
    OP = {
        0: operator.eq,  # "=="
        -1: operator.le,  # "<="
        -2: operator.lt,  # "<"
        1: operator.ge,  # ">="
        2: operator.gt,  # ">"
    }
    OP_DOMAIN = [-2, -1, 0, 1, 2]