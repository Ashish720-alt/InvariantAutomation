""" Hyper Parameters.
"""
import operator
from math import pi, e

class Configure:
    
    dspace_radius = 1000000
    dspace_intmin = -1 * dspace_radius
    dspace_intmax = 1 * dspace_radius

    num_processes = 4 #Number of threads

    maxSArun = 10000000

    k = 2
    r = 1

    BoxesCountSmallSpace = 50
    probenet_success = 0.8

    s = 5 #Number of cex Z3 generates

    translation_range = 10
    rotation_degree = pi/4 

    p_rot = 0.5

    gamma = e #Need floating point constant here
    beta0 = 100 #2^-20 ~ 10^(-7)
    beta = 1
    # descent_prob = 0.5

    # d = 100

    temp_C = 1000000.0 #Change this!!

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
    SAMPLEPOINTS_DEBUGGER_WINDOW = 1000
    PRINT_ITERATIONS = ON
    PRETTYPRINTINVARIANT_ITERATIONS = ON
    PRINT_REJECT_ITERATIONS = ON
    PRINT_COSTLIST = OFF
    PRINT_Z3_ITERATIONS = ON
    PRINT_TIME_STATISTICS = ON
    NUM_ROUND_CHECK_EARLY_EXIT = 100
    PRINT_COLORED_THREADS = ON
    

    # n2Plotter hyperparameters
    n2PLOTTER_LOW_RES = 300
    n2PLOTTER_HIGH_RES = 1000
    n2PLOTTER_LARGESCALE = 1
    n2PLOTTER_LARGESCALEINTERVAL = [dspace_intmin - 250,dspace_intmax +250]
    n2PLOTTER_SMALLSCALE = 0
    n2PLOTTER_SMALLSCALEINTERVAL = [-50,50]
    n2PLOTTER_CUSTOMIZEDSCALE = -1
    n2PLOTTER_CUSTOMIZEDSCALEINTERVAL = [0, 0] #Insert Required Interval