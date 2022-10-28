""" Hyper Parameters.
"""
import operator

class Configure:
    
    dspace_intmin = -10000
    dspace_intmax = 10000
  
    k = 2
    r = 1

    s = 5

    p = 0.0005 #if you take this as 0.05, this rejects many good invariant jumps.
    p_rot = 0.5

    alpha = 10**7
    gamma = 2.0 #Need floating point constant here
    beta0 = 20 #2^-20 ~ 10^(-7)
    descent_prob = 0.5

    d = 100
    centres_sampled = 20
    centre_walklength = 10

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
    SAMPLEPOINTS_DEBUGGER = OFF
    PRINT_ITERATIONS = ON
    PRETTYPRINTINVARIANT_ITERATIONS = OFF
    PRINT_REJECT_ITERATIONS = OFF
    PRINT_STAY_ITERATIONS = ON
    PRINT_Z3_ITERATIONS = ON
    PRINT_TIME_STATISTICS = ON
