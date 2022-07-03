
""" Hyper Parameters.
"""
import operator

class Configure:
    
    dspace_intmin = -10000
    dspace_intmax = 10000
  
    
    k = 2
    r = 1

    s = 100

    
    # cost function related
    NUM_COUNTEREXAMPLE = 10 
    K1 = 1
    K2 = 2
    K3 = 1




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

