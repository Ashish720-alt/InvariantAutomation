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

def statistics(t, I, cost, mincost, descent):
    I_list = DNF_aslist(I)
    if (conf.PRINT_ITERATIONS == conf.ON):
        descent_string = "(L)" if descent else ""
        print("t = ", t, ":\t", I_list , "\t", "(cost, mincost) = ", (cost, mincost) , "\t", descent_string )
    

def z3statistics(correct, original_samplepoints, added_samplepoints, z3_callcount, timeout):
    if (conf.PRINT_ITERATIONS == conf.ON):    
        print("Z3 Call " + str(z3_callcount) + ":\n", "\tTimeout = ", timeout, '\n', "\tz3_correct = ", correct)
        prettyprint_samplepoints(original_samplepoints, "original-selection-points", "\t")
        prettyprint_samplepoints(added_samplepoints, "CEX-generated", "\t")

def invariantfound(I):
    print("Invariant Found:\t", end = '')
    prettyprint_invariant(I)
