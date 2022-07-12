from configure import Configure as conf
from dnfs_and_transitions import DNF_aslist
from z3verifier import DNF_to_z3expr

def prettyprint_samplepoints(samplepoints, header, indent):
    print(indent + header + ":")
    print(2*indent + "+ := ", end = '')
    for plus in samplepoints[0]:
        print(plus,',\t', end = '')
    print("\n" + 2*indent + "- := ", end = '')
    for minus in samplepoints[1]:
        print(minus,',\t', end = '')        
    print("\n" + 2*indent + "-> := ", end = '')
    for ICE in samplepoints[2]:
        print(ICE,',\t', end = '')    
    print("\n", end = '')

def prettyprint_invariant(I):
    print(DNF_to_z3expr(I, primed = 0))

def initialized():
    if (conf.PRINT_ITERATIONS == conf.ON):
        print("Initialization Complete...")

def statistics(t, I, cost, mincost, descent, reject):
    I_list = DNF_aslist(I)
    if (conf.PRINT_ITERATIONS == conf.ON):
        if (reject):
            if (conf.PRINT_REJECT_ITERATIONS == conf.ON):
                end_string = "[X]" 
                print("t = ", t, ":\t", I_list , "\t", "(f, cost) = ", (cost, mincost) , "\t", end_string )
                return
            else:
                return
        else:
            end_string = "(L)" if descent else ""
            print("t = ", t, ":\t", I_list , "\t", "(f, cost) = ", (cost, mincost) , "\t", end_string )
            return
    

def z3statistics(correct, original_samplepoints, added_samplepoints, z3_callcount, timeout):
    if (conf.PRINT_ITERATIONS == conf.ON):    
        print("Z3 Call " + str(z3_callcount) + ":\n", "\tTimeout = ", int(timeout), '\n', "\tz3_correct = ", correct)
        prettyprint_samplepoints(original_samplepoints, "original-selection-points", "\t")
        prettyprint_samplepoints(added_samplepoints, "CEX-generated", "\t")

def invariantfound(I):
    print("Invariant Found:\t", end = '')
    prettyprint_invariant(I)

def timestatistics(neighbor_time , rest_time, total_iterations, z3_time, initialize_time, z3_callcount ):
    if (conf.PRINT_TIME_STATISTICS == conf.ON): 
        print("\nTime Statistics:")
        print("\tTotal Initialization and Re-initialization Time: ", initialize_time)
        print("\tTotal Neighbor Time: ", neighbor_time)
        print("\tTotal remaining MCMC loop Time: ", rest_time)
        print("\tTotal Z3 Time: ", z3_time)
        print("\tTotal MCMC iterations: ", total_iterations)
        print("\tTotal Z3 calls: ", z3_callcount)
