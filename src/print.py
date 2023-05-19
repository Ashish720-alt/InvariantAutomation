from configure import Configure as conf
from dnfs_and_transitions import RTI_to_LII, DNF_aslist,  list3D_to_listof2Darrays, dnfTrue, dnfFalse
from z3verifier import DNF_to_z3expr
from colorama import Fore, Back, Style
from math import floor

def print_colorslist(t):
    if (conf.PRINT_COLORED_THREADS == conf.ON):
        temp = [Fore.RED, Fore.BLUE, Fore.LIGHTGREEN_EX,  Fore.YELLOW, Fore.BLACK,  Fore.CYAN, Fore.LIGHTBLACK_EX, Fore.LIGHTBLUE_EX, Fore.LIGHTCYAN_EX,  Fore.LIGHTMAGENTA_EX, 
            Fore.LIGHTRED_EX, Fore.LIGHTWHITE_EX, Fore.MAGENTA,  Fore.RESET, Fore.WHITE]
    else:
        temp = [Fore.WHITE]
    templen = len(temp)
    q = floor(t / templen)
    r = t - (q * templen)
    return temp * q + temp[:r]


def decimaltruncate(number, digits = 7):
    if (digits == -1):
        return number
    power = "{:e}".format(number).split('e')[1]
    return round(number, -(int(power) - digits))

def decimaltruncate_list(l, digits = 4):
    return [decimaltruncate(x, digits) for x in l ]

def samplepoints_debugger(t, samplepoints):
    if (conf.SAMPLEPOINTS_DEBUGGER == conf.ON):
        if (t % 1000 == 0):
            prettyprint_samplepoints(samplepoints, "Samplepoints Now", "\t")    

def prettyprint_samplepoints(samplepoints, header, indent):
    print(indent + header + ":")
    print(2*indent + "+ := ", end = '')
    for plus in samplepoints[0]:
        print(plus,' , ', end = '')
    print("\n" + 2*indent + "- := ", end = '')
    for minus in samplepoints[1]:
        print(minus,' , ', end = '')        
    print("\n" + 2*indent + "-> := ", end = '')
    for ICE in samplepoints[2]:
        print('(', ICE[0], '->', ICE[1], ')',' , ', end = '')    
    print("\n", end = '')



def prettyprint_invariant(I, endstring , Vars):
    n = len(I[0][0]) - 2
    rv = ""
    opstring_dict = { -2 : "<" , -1 : "<=" , 0 : "==" , 1 : ">=" , 2 : ">" }
    if (I != []):
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
                        # rv = rv + str(p[i]) + "*" + ("x%s" % i) #For the no variable print
                        rv = rv + str(p[i]) + "*" + (Vars[i])
                    elif (i == n):
                        rv = rv + " " + opstring_dict[p[i]] + " "
                    else:
                        rv = rv + str(p[i])
                    if (i < n - 1):
                        rv = rv + " + "
                rv = rv + ')'
            rv = rv + ')'
    rv = rv + endstring
    return rv

    
    # print(DNF_to_z3expr(I, primed = 0))

def SAsuccess(process_id, colorslist):
    if (conf.PRINT_ITERATIONS == conf.ON):
        print(colorslist[process_id] + "Process ", process_id, " found approximate invariant!")
        print(Style.RESET_ALL)    

def SAexit(process_id, colorslist):
    if (conf.PRINT_ITERATIONS == conf.ON):
        print(colorslist[process_id] + "Process ", process_id, " exit early!")
        print(Style.RESET_ALL)    

def SAfail(process_id, colorslist):
    if (conf.PRINT_ITERATIONS == conf.ON):
        print(colorslist[process_id] + "Process ", process_id, " failed!")
        print(Style.RESET_ALL, end = '')    

def initialized(A, B, Vars):
    if (conf.PRINT_ITERATIONS == conf.ON):
        print("Initialization Complete...")
        print("\tAffine SubSpace: ", end = '')
        if (A != []):
            print( prettyprint_invariant(A, '', Vars))
        else:
            print( '\n', end = '')
        print("\tNon Iterative Precondition: ", end = '')
        if (B != []):
            print( prettyprint_invariant(B, '', Vars))
        else:
            print( '\n')

def statistics(p, t, I, mincost, descent, reject, costlist, acc, Vars, colorslist):
    if (conf.PRINT_ITERATIONS == conf.ON):    
        if (reject):
            if (conf.PRINT_REJECT_ITERATIONS == conf.ON):
                end_string = "[X]" 
                if (conf.PRETTYPRINTINVARIANT_ITERATIONS == conf.OFF):
                    print(colorslist[p] + "P = ", p, ",", "t = ", t,":", "\t", I, "\t", "(cost, a) = ", ( decimaltruncate(mincost ), 
                            decimaltruncate(acc )) , "\t", end_string )
                if (conf.PRINT_COSTLIST == conf.ON):
                    print(decimaltruncate_list(costlist), "\n")
                else:
                    print(colorslist[p] + "P = ", p, ",", "t = ", t,":", "\t", prettyprint_invariant((list3D_to_listof2Darrays(I)), '', Vars), "\t", "(cost, a) = ", ( 
                            decimaltruncate(mincost ), decimaltruncate(acc )) , "\t", end_string )
                if (conf.PRINT_COSTLIST == conf.ON):
                    print(decimaltruncate_list(costlist), "\n")
                print(Style.RESET_ALL, end = '')
            else:
                return
        else:
            end_string = "(L)" if descent else "   "
            if (conf.PRETTYPRINTINVARIANT_ITERATIONS == conf.OFF):
                print(colorslist[p] + "P = ", p, ",", "t = ", t,":", "\t", I, "\t", "(cost, a) = ", (decimaltruncate(mincost ),  
                        decimaltruncate(acc )) , "\t", end_string )
                if (conf.PRINT_COSTLIST == conf.ON):
                    print(decimaltruncate_list(costlist), "\n")
            else:
                print(colorslist[p] + "P = ", p, ",", "t = ", t,":", "\t", prettyprint_invariant((list3D_to_listof2Darrays(I)), '',  Vars), "\t", "(cost, a) = ", ( 
                        decimaltruncate(mincost ), decimaltruncate(acc )) , "\t", end_string )
                if (conf.PRINT_COSTLIST == conf.ON):
                    print(decimaltruncate_list(costlist), "\n")
            print(Style.RESET_ALL, end = '')
            return
    return 
    

def z3statistics(correct, original_samplepoints, added_samplepoints, z3_callcount, timeout):
    if (conf.PRINT_Z3_ITERATIONS == conf.ON):    
        print("Z3 Call " + str(z3_callcount) + ":\n", "\tTimeout = ", int(timeout), '\n', "\tz3_correct = ", correct)
        prettyprint_samplepoints(original_samplepoints, "original-selection-points", "\t")
        prettyprint_samplepoints(added_samplepoints, "CEX-generated", "\t")
        print("\n\n")

def invariantfound( NonIterativeI , Affine_I , I, Vars):
    print("Invariant Found:\t", end = '')
    n = len(Vars) 
    if (NonIterativeI != []):
        print(prettyprint_invariant( list3D_to_listof2Darrays(NonIterativeI), '  \/  [', Vars), end = '')
    if (Affine_I != []):
        print(prettyprint_invariant( list3D_to_listof2Darrays(Affine_I), '  /\  [', Vars),  end = '')
    print(prettyprint_invariant( list3D_to_listof2Darrays(I), '', Vars), end = '')
    if (Affine_I != []):
        print(']', end = '')
    if (NonIterativeI != []):
        print(']', end = '')
    print('\n')

def noInvariantFound (Z3calls):
    print("SA failed to converge after", Z3calls , "Z3 runs")

def timestatistics(mcmc_time, total_iterations, z3_time, initialize_time, z3_callcount, threads ):
    if (conf.PRINT_TIME_STATISTICS == conf.ON): 
        print("\nTime Statistics:")
        print("\tTotal Initialization and Re-initialization Time: ", initialize_time)
        print("\tTotal MCMC time: ", mcmc_time)
        print("\tTotal Z3 Time: ", z3_time)
        print("\tTotal MCMC iterations: ", total_iterations)
        print("\tTotal Z3 calls: ", z3_callcount)
        print("\tNumber of Threads: ", threads)
