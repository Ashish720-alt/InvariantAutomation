""" Hyper Parameters.
"""
import operator
from math import pi, e

class Configure:
    ON = 1
    OFF = 0

    #CHANGE THIS FOR SERVER RUNS!!!
    REMOTE_SERVER = OFF
    REMOTE_PATH = "../../rotation-graph/"

    '''Invariant Space hyperparameters'''
    coeff_k = 2
    coeff_r = 1

    '''State Space hyperparameters'''
    dspace_full = ON #Change this
    dspace_maxradius = 1.5 * 1000000
    dspace_scaledradius = 100 #Change testing Radius
    dspace_radius = dspace_maxradius if (dspace_full == ON) else dspace_scaledradius 
    dspace_intmin = -1 * dspace_radius
    dspace_intmax = 1 * dspace_radius

    '''SA hyperparameters '''
    NUM_ROUND_CHECK_EARLY_EXIT = 100
    translation_range = 5
    rotation_degree = 5
    rotation_rad = rotation_degree * (pi/180) 
    rotation_probability = 0.5
    num_processes = 4 #Number of threads
    t0 = 2
    maxSArun =  1000000 #1000000
    I0_samples = 100 #Initial Invariant Samples.
    I0_samples_n1 = 1000
    
    #Search Space Graph Choices
    BASIC_ROTATION = ON
    BASIC_ROTATION_k0 = 2
    COR_SIMPLIFIED = ON
    TRANSLATION_SMALL = ON
    ONLY_ROTATION = -1
    ONLY_TRANSLATION = 1
    ROTATION_AND_TRANSLATION = 0
    GUESS_SCHEME = ROTATION_AND_TRANSLATION

    #Cost hypeparameters
    COST_ILP = OFF
    COST_DISTANCE_UNNORMALIZED = OFF
    COST_DISTANCE_NORMALIZED = ON
    costnormalizer_K = 1000.0
    costnormalizer_m = 20.0    
    COST_DATASET_NORMALIZER = ON

    #Experimental T0 parameters
    T0_X0 = 0.30 #Expected acceptance ratio
    T0_e = 0.001 #how much error in T0_X0 is allowed when computing T0
    T0_COSTLISTLENGTH = 100
    T0_rotationMaxProb = 0.5
    
    #Stagnant Runs Debugger
    CHECK_LOCALMINIMA = OFF
    STAGNANT_TIME_WINDOW = 1000
    STAGNANT_COST_RANGE = 1
    LOCAL_MINIMA_DEPTH_CHECKER = 1000

    #Average Acceptance Probability Checker
    AVERAGE_ACC_PROB_CHECKER = OFF
    AVERAGE_ACC_PROB_WINDOW = 500

    ''' Datapoint Sampling hyperparameters '''
    #Iterated Implication pairs
    maxiterateImplPair = 100 
    maxiterateICE = 20
    
    #e-net parameters:
    e0 = 0.5
    probenet_success = 0.8
    z3_stepwindow = 10

    #z3 parameters
    s = 5 
    z3_C1_Tmax = [0]
    z3_C2_Tmax = [100, 0]
    z3_C3_Tmax = [100, 0]

    # Operators
    OP = {
        0: operator.eq,  # "=="
        -1: operator.le,  # "<="
        -2: operator.lt,  # "<"
        1: operator.ge,  # ">="
        2: operator.gt,  # ">"
        10: operator.ne, # "!="
    }



    '''Print iterations hyperparameters'''
    TERMINAL = 1
    FILE = -1
    TERMINAL_AND_FILE = 0
    PRINTING_MODE = FILE 
    
    PRINT_ITERATIONS = OFF
    PRINT_REJECT_ITERATIONS = OFF
    PRINT_COSTLIST = OFF
    PRINT_Z3_ITERATIONS = OFF
    PRINT_TIME_STATISTICS = ON
    PRINT_COLORED_THREADS = ON
    

    '''Graph Plotter hyperparameters '''
    #CostvsTimePlotter
    COST_PLOTTER = OFF
    
    #InvariantSpacePlots
    INVARIANTSPACE_PLOTTER = OFF
    INVARIANTSPACE_MAXCONST = 3

    # Data Space Plots (for n = 2):  
    n2PLOTTER = OFF
    n2PLOTTER_WINDOW = 500
    n2PLOTTER_LOW_RES = 200
    n2PLOTTER_HIGH_RES = 1000
    n2PLOTTER_LARGESCALE = 1
    n2PLOTTER_LARGESCALEINTERVAL = [dspace_intmin - 250,dspace_intmax +250]
    n2PLOTTER_SMALLSCALE = 0
    n2PLOTTER_SMALLSCALEINTERVAL = [-50,50]
    n2PLOTTER_CUSTOMIZEDSCALE = -1
    n2PLOTTER_CUSTOMIZEDSCALEINTERVAL = [0, 0] #Insert Required Interval
    
    ''' Optimizations hyperparameters'''
    NONITERATIVE_PRECONDITION = OFF
    AFFINE_SPACES = OFF
    # Implement terminating inductive invariant?
