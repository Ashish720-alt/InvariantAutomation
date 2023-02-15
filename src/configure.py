""" Hyper Parameters.
"""
import operator
from math import pi

class Configure:
    
    dspace_intmin = -1000000
    dspace_intmax = 1000000

    threads = 1

    k = 2
    r = 1

    ICEBallRadius = 1

    s = 5 #Number of cex Z3 generates

    translation_range = 10
    rotation_degree = pi/4 + 0.01

    p_rot = 0.5

    alpha = 10**7
    gamma = 2.0 #Need floating point constant here
    beta0 = 20 #2^-20 ~ 10^(-7)
    descent_prob = 0.5

    d = 100
    centres_sampled = 20
    # centre_walklength = 10

    temp_C = 100.0

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
    PRETTYPRINTINVARIANT_ITERATIONS = ON
    PRINT_REJECT_ITERATIONS = OFF
    PRINT_COSTLIST = OFF
    PRINT_Z3_ITERATIONS = ON
    PRINT_TIME_STATISTICS = ON
