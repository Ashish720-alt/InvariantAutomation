
""" Hyper Parameters.
"""
import operator

class Configure:
    
    dspace_intmin = -10000
    dspace_intmax = 10000
  
    k = 2
    r = 1

    s = 20

    p = 0.0005 #if you take this as 0.05, this rejects many good invariant jumps.
    
    alpha = 1e+10
    gamma = 2.0 #Need floating point constant here
    beta0 = 100 #10 doesnt work for gamma = 2.0 or 10.0 as it takes too many bad transitions.

    # Operators
    OP = {
        0: operator.eq,  # "=="
        -1: operator.le,  # "<="
        -2: operator.lt,  # "<"
        1: operator.ge,  # ">="
        2: operator.gt,  # ">"
        10: operator.ne, # "!="
    }

    ON = 1
    OFF = 0
    # Print iterations or not?
    PRINT_ITERATIONS = ON
    PRINT_REJECT_ITERATIONS = ON
