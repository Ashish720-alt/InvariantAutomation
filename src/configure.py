""" Hyper Parameters.
"""
import operator
from math import pi, e

class Configure:
    
    dspace_intmin = -1000000
    dspace_intmax = 1000000

    num_processes = 3

    maxSArun = 10000000

    k = 2
    r = 1

    ICEBallRadius = 1

    s = 5 #Number of cex Z3 generates

    translation_range = 10
    rotation_degree = pi/4 

    p_rot = 0.5

    gamma = e #Need floating point constant here
    beta0 = 100 #2^-20 ~ 10^(-7)
    beta = 100
    # descent_prob = 0.5

    # d = 100

    temp_C = 100.0 #Change this!!

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
    NUM_ROUND_CHECK_EARLY_EXIT = 1000