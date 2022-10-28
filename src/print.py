from configure import Configure as conf
from dnfs_and_transitions import RTI_to_LII, DNF_aslist,  list3D_to_listof2Darrays
from z3verifier import DNF_to_z3expr

def decimaltruncate(number, digits = 4):
    if (digits == -1):
        return number
    power = "{:e}".format(number).split('e')[1]
    return round(number, -(int(power) - digits))

def decimaltruncate_list(l, digits = 4):
    return [decimaltruncate(x, digits) for x in l ]

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



def prettyprint_invariant(I, endstring):
    n = len(I[0][0]) - 2
    rv = ""
    for k1,cc in enumerate(I):
        if (k1 > 0):
            rv = rv + " \/ " 
        rv = rv + "("
        for k2,p in enumerate(cc):
            if (k2 > 0):
                rv = rv + " /\ "
            rv = rv + "("
            for i in range(len(p)):
                if (i < n):
                    rv = rv + str(p[i]) + ("*x%s" % i)
                elif (i == n):
                    rv = rv + " <= "
                else:
                    rv = rv + str(p[i])
                if (i < n - 1):
                    rv = rv + " + "
            rv = rv + ')'
        rv = rv + ')'
    rv = rv + endstring
    return rv

    
    print(DNF_to_z3expr(I, primed = 0))

def initialized():
    if (conf.PRINT_ITERATIONS == conf.ON):
        print("Initialization Complete...")

def statistics(t, is_change, I, cost, mincost, descent, reject, costlist, acc):
    if (conf.PRINT_ITERATIONS == conf.ON):
        if (not is_change):
            if (conf.PRINT_STAY_ITERATIONS == conf.ON):
                end_string = "[S]" 
                if (conf.PRETTYPRINTINVARIANT_ITERATIONS == conf.OFF):
                    print("t = ", t, "\t", I, "\t", "(f, cost, a) = ", (decimaltruncate(cost ), decimaltruncate(mincost ), decimaltruncate(acc )) 
                                    , "\t", end_string, "\t", decimaltruncate_list(costlist), "\n" )
                else:
                    print("t = ", t, "\t", prettyprint_invariant((list3D_to_listof2Darrays(I)), ''), "\t", "(f, cost, a) = ", (decimaltruncate(cost ), 
                            decimaltruncate(mincost ), decimaltruncate(acc )) , "\t", end_string, "\t", decimaltruncate_list(costlist), "\n" )
                return
            else:
                return            
        else:
            if (reject):
                if (conf.PRINT_REJECT_ITERATIONS == conf.ON):
                    end_string = "[X]" 
                    if (conf.PRETTYPRINTINVARIANT_ITERATIONS == conf.OFF):
                        print("t = ", t, "\t", I, "\t", "(f, cost, a) = ", (decimaltruncate(cost ), decimaltruncate(mincost ), 
                                decimaltruncate(acc )) , "\t", end_string, "\t", decimaltruncate_list(costlist), "\n" )
                    else:
                        print("t = ", t, "\t", prettyprint_invariant((list3D_to_listof2Darrays(I)), ''), "\t", "(f, cost, a) = ", (decimaltruncate(cost ), 
                                decimaltruncate(mincost ), decimaltruncate(acc )) , "\t", end_string, "\t", decimaltruncate_list(costlist), "\n")
                else:
                    return
            else:
                end_string = "(L)" if descent else "   "
                if (conf.PRETTYPRINTINVARIANT_ITERATIONS == conf.OFF):
                    print("t = ", t, "\t", I, "\t", "(f, cost, a) = ", (decimaltruncate(cost ), decimaltruncate(mincost ), 
                            decimaltruncate(acc )) , "\t", end_string, "\t", decimaltruncate_list(costlist), "\n" )
                else:
                    print("t = ", t, "\t", prettyprint_invariant((list3D_to_listof2Darrays(I)), ''), "\t", "(f, cost, a) = ", (decimaltruncate(cost ), 
                            decimaltruncate(mincost ), decimaltruncate(acc )) , "\t", end_string, "\t", decimaltruncate_list(costlist) , "\n")
                return
    

def z3statistics(correct, original_samplepoints, added_samplepoints, z3_callcount, timeout):
    if (conf.PRINT_Z3_ITERATIONS == conf.ON):    
        print("Z3 Call " + str(z3_callcount) + ":\n", "\tTimeout = ", int(timeout), '\n', "\tz3_correct = ", correct)
        prettyprint_samplepoints(original_samplepoints, "original-selection-points", "\t")
        prettyprint_samplepoints(added_samplepoints, "CEX-generated", "\t")
        print("\n\n")

def invariantfound(I):
    print("Invariant Found:\t", end = '')
    print(prettyprint_invariant( list3D_to_listof2Darrays(I), ''))

def timestatistics(mcmc_time, total_iterations, z3_time, initialize_time, z3_callcount ):
    if (conf.PRINT_TIME_STATISTICS == conf.ON): 
        print("\nTime Statistics:")
        print("\tTotal Initialization and Re-initialization Time: ", initialize_time)
        print("\tTotal MCMC time: ", mcmc_time)
        print("\tTotal Z3 Time: ", z3_time)
        print("\tTotal MCMC iterations: ", total_iterations)
        print("\tTotal Z3 calls: ", z3_callcount)
