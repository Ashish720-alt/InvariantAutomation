
""" Hyper Parameters.
"""
import operator

class Configure:
    s = 10

    
    # cost function related
    NUM_COUNTEREXAMPLE = 10 
    
    # Fact clause -> -1, Inductive clause -> 0, Query Clause -> 1
    K = {
        -1: 1,
        0: 2,
        1: 1
    }

    SB = 5
    SD = 3

    max_guesses = 10000

    max_disjuncts = 5
    max_conjuncts = 5


    ON = 1
    OFF = 0
    # Print warnings or not?
    DISPLAY_WARNINGS = ON
    # Print iterations or not?
    PRINT_ITERATIONS = ON

    # Operators
    OP = {
        0: operator.eq,  # "=="
        -1: operator.le,  # "<="
        -2: operator.lt,  # "<"
        1: operator.ge,  # ">="
        2: operator.gt,  # ">"
        10: operator.ne, # "!="
        -10: operator.ne, # "!="
    }
    # Don't add special operators to operator domain
    OP_DOMAIN = [-2, -1, 0, 1, 2]