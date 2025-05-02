from configure import Configure as conf
from dnfs_and_transitions import RTI_to_LII, DNF_aslist,  list3D_to_listof2Darrays, dnfTrue, dnfFalse
from z3verifier import DNF_to_z3expr
from colorama import Fore, Back, Style
from math import floor, isnan, isinf
from n2Invariantplotter import do_plot
import multiprocessing

lock = multiprocessing.Lock()


def printBenchmarkName( inputname, outputfile):
    print_with_mode(Fore.WHITE, "Working on " + inputname, endstr = '\n', file = outputfile)

def print_with_mode(color, s, endstr = '\n', file = None):
    lock.acquire()
    if (conf.PRINTING_MODE == conf.TERMINAL or conf.PRINTING_MODE == conf.TERMINAL_AND_FILE):
        print(color + s, end = endstr)
        print(Style.RESET_ALL, end = '')  
    if (conf.PRINTING_MODE == conf.FILE or conf.PRINTING_MODE == conf.TERMINAL_AND_FILE or conf.PRINTING_MODE == conf.SINGLE_FILE_ALL_PROGRAMS ):
        file.write(s + endstr)    
    lock.release()
    return


def print_colorslist(t):
    if (conf.PRINT_COLORED_THREADS == conf.ON):
        temp = [Fore.WHITE, Fore.RED, Fore.BLUE, Fore.LIGHTGREEN_EX,  Fore.YELLOW, Fore.BLACK,  Fore.CYAN, Fore.LIGHTBLACK_EX, Fore.LIGHTBLUE_EX, Fore.LIGHTCYAN_EX,  Fore.LIGHTMAGENTA_EX, 
            Fore.LIGHTRED_EX, Fore.LIGHTWHITE_EX, Fore.MAGENTA,  Fore.RESET]
    else:
        temp = [Fore.WHITE]
    templen = len(temp)
    q = floor(t / templen)
    r = t - (q * templen)
    return temp * q + temp[:r]

def list_to_string(L):
    rv = "["
    for i in range(len(L)):
        rv = rv + str(L[i])
        if (i != len(L) - 1):
            rv = rv + ", "
    rv = rv + " ]"
    return rv

def decimaltruncate(number, digits = 7):
    if (digits == -1):
        return number
    power = int("{:e}".format(number).split('e')[1])
    if isnan(power) or isinf(power): #To check for NAN warnings!
        return number
    else:
        return round(number, -(power - digits))

def decimaltruncate_list(l, digits = 4):
    return [decimaltruncate(x, digits) for x in l ]

def n2plotter(inputname, n, p, z3calls, t, samplepoints, I, colorslist, outputfile = None):
    if (t % conf.n2PLOTTER_WINDOW == 0):
        if (n == 2):
            print_with_mode(colorslist[p],'P' + str(p) + ' plotting graph...', file = outputfile)
            do_plot(inputname + '_Z3:' + str(z3calls) + 'P:' + str(p) + 'T:' + str(t) + '(LargeScale).png', 
                    '2DStateSpacePlots', conf.n2PLOTTER_LARGESCALE, I, samplepoints, resolution = conf.n2PLOTTER_HIGH_RES)
            do_plot(inputname + '_Z3:' + str(z3calls) + 'P:' + str(p) + 'T:' + str(t) + '(SmallScale).png', 
                    '2DStateSpacePlots', conf.n2PLOTTER_SMALLSCALE, I, samplepoints, resolution = conf.n2PLOTTER_HIGH_RES)            
        else:    
            prettyprint_samplepoints(samplepoints, "Samplepoints Now", "\t", outputfile)    

def prettyprint_samplepoints(samplepoints, header, indent, outputfile = None):
    print_with_mode(Fore.WHITE, indent + header + ":", file = outputfile)
    print_with_mode(Fore.WHITE, 2*indent + "+ := ", endstr = '', file = outputfile)
    for plus in samplepoints[0]:
        print_with_mode(Fore.WHITE, list_to_string(plus) + ' , ', endstr = '', file = outputfile)
    print_with_mode(Fore.WHITE, "\n" + 2*indent + "- := ", endstr = '', file =  outputfile)
    for minus in samplepoints[1]:
        print_with_mode(Fore.WHITE, list_to_string(minus) + ' , ', endstr = '', file = outputfile)        
    print_with_mode(Fore.WHITE, "\n" + 2*indent + "-> := ", endstr = '', file = outputfile)
    for ICE in samplepoints[2]:
        print_with_mode(Fore.WHITE, '( ' + list_to_string(ICE[0]) + ' -> ' + list_to_string(ICE[1]) + ' )  , ', endstr = '', file = outputfile)    
    print_with_mode(Fore.WHITE, "\n", endstr = '', file = outputfile)
    print_with_mode(Fore.WHITE, '\n', file = outputfile)
    return

def prettyprint_ICEpairs(ImplicationPairs, header, indent, outputfile = None):
    print_with_mode(Fore.WHITE, indent + header + ":", file = outputfile) 
    print_with_mode(Fore.WHITE, 2*indent + "-> := ", endstr = '', file = outputfile)
    for ICE in ImplicationPairs:
        print_with_mode(Fore.WHITE, '( ' + list_to_string(ICE[0]) + ' -> ' + list_to_string(ICE[1]) + ' )  , ', endstr = '', file = outputfile)    
    print_with_mode(Fore.WHITE, "\n", endstr = '', file = outputfile)
    return
    
    
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

def SAsuccess(process_id, colorslist, outputfile):
    if (conf.PRINT_ITERATIONS == conf.ON):
        print_with_mode(colorslist[process_id], "Process " + str(process_id) + " found approximate invariant!", endstr = '\n', file = outputfile)
          

def SAexit(process_id, colorslist, outputfile):
    if (conf.PRINT_ITERATIONS == conf.ON):
        print_with_mode(colorslist[process_id], "Process " +  str(process_id) + " exit early!",endstr = '\n', file = outputfile) 

def SAfail(process_id, colorslist, outputfile):
    if (conf.PRINT_ITERATIONS == conf.ON):
        print_with_mode(colorslist[process_id], "Process " +  str(process_id) + " failed!",endstr = '\n', file = outputfile)  

def initialized(A, B, Vars, outputfile):
    if (conf.PRINT_ITERATIONS == conf.ON):
        print_with_mode(Fore.WHITE, "Initialization Complete...", file = outputfile)
        print_with_mode(Fore.WHITE, "\tAffine SubSpace: ", endstr = '', file = outputfile)
        if (A != []):
            print_with_mode(Fore.WHITE, prettyprint_invariant(A, '', Vars), file = outputfile)
        else:
            print_with_mode(Fore.WHITE, '\n', endstr = '', file = outputfile)
        print_with_mode(Fore.WHITE, "\tNon Iterative Precondition: ", endstr = '', file = outputfile)
        if (B != []):
            print_with_mode(Fore.WHITE, prettyprint_invariant(B, '', Vars), file = outputfile)
        else:
            print_with_mode(Fore.WHITE, '\n', file = outputfile)

def printTemperaturePrompt(colorslist, outputfile):
    if (conf.PRINT_ITERATIONS == conf.ON):
        print_with_mode(colorslist[0], "Calculating initial temp ...", endstr= '\n', file = outputfile)
    return

# Fix the printing
def statistics(p, t, I, mincost, descent, reject, costlist, acc, Vars, colorslist, outputfile):
    if (conf.PRINT_ITERATIONS == conf.ON):    
        if (reject):
            if (conf.PRINT_REJECT_ITERATIONS == conf.ON):
                end_string = "[X]" 
                print_with_mode(colorslist[p], "P = " + str(p) + "," + "t = " + str(t) + ":" + "\t" + prettyprint_invariant((list3D_to_listof2Darrays(I)), '', Vars) + "\t" + "(cost, a) = " + '(' 
                         + str(decimaltruncate(mincost )) + "," + str(decimaltruncate(acc )) + ')' + "\t" + end_string, endstr= '\n', file = outputfile )
                if (conf.PRINT_COSTLIST == conf.ON):
                    print_with_mode(Fore.WHITE, list_to_string( decimaltruncate_list(costlist)), endstr = "\n", file = outputfile )
            else:
                return
        else:
            end_string = "(L)" if descent else "   "
            print_with_mode(colorslist[p], "P = " + str(p) + "," + "t = " + str(t) + ":" + "\t" + prettyprint_invariant((list3D_to_listof2Darrays(I)), '',  Vars) + "\t" + "(cost, a) = " + '(' 
                    + str(decimaltruncate(mincost )) + "," + str(decimaltruncate(acc )) + ')' + "\t" + end_string, endstr= '\n', file = outputfile )
            if (conf.PRINT_COSTLIST == conf.ON):
                print_with_mode(Fore.WHITE, list_to_string( decimaltruncate_list(costlist)), endstr = "\n", file = outputfile )
            return
    return 
    

def z3statistics(correct, original_samplepoints, added_samplepoints, z3_callcount, timeout, new_enet, e , eNetPoints, iteratedICEpairs, outputf):
    if (conf.PRINT_Z3_ITERATIONS == conf.ON):    
        print_with_mode(Fore.WHITE, "Z3 Call " + str(z3_callcount) + ":\n" + "\tTimeout = " + str(int(timeout)) + '\n' + "\tz3_correct = " + str(correct) + '\n' + "\te value = " + str(e), endstr = '\n', file = outputf)
        prettyprint_samplepoints(original_samplepoints, "Original-selection-points", "\t", outputfile = outputf)
        if(new_enet):
            prettyprint_samplepoints(eNetPoints, "\nAdded-selection-points", "\t", outputfile = outputf)
        prettyprint_samplepoints(added_samplepoints, "CEX-generated", "\t", outputfile = outputf)
        prettyprint_ICEpairs(iteratedICEpairs, "Iterated ICE pairs", "\t", outputfile = outputf)
        print_with_mode(Fore.WHITE, "\n\n", endstr= '\n', file= outputf)

def invariantfound( NonIterativeI , Affine_I , I, Vars, outputfile):
    print_with_mode(Fore.WHITE, "Invariant Found:\t", endstr = '', file = outputfile)
    n = len(Vars) 
    if (NonIterativeI != []):
        print_with_mode(Fore.WHITE, prettyprint_invariant( list3D_to_listof2Darrays(NonIterativeI), '  \/  [', Vars), endstr = '', file = outputfile)
    if (Affine_I != []):
        print_with_mode(Fore.WHITE, prettyprint_invariant( list3D_to_listof2Darrays(Affine_I), '  /\  [', Vars),  endstr = '', file = outputfile)
    print_with_mode(Fore.WHITE, prettyprint_invariant( list3D_to_listof2Darrays(I), '', Vars), endstr = '', file = outputfile)
    if (Affine_I != []):
        print_with_mode(Fore.WHITE, ']', endstr = '', file = outputfile)
    if (NonIterativeI != []):
        print_with_mode(Fore.WHITE, ']', endstr = '', file = outputfile)
    print_with_mode(Fore.WHITE, '\n', file = outputfile)

def noInvariantFound (Z3calls, outputfile):
    print_with_mode(Fore.WHITE, "All SA threads failed to converge after" +  str(Z3calls) + "Z3 runs", file = outputfile)

def timestatistics(mcmc_time, total_iterations, z3_time, initialize_time, z3_callcount, threads, outputfile ):
    if (conf.PRINT_TIME_STATISTICS == conf.ON): 
        print_with_mode(Fore.WHITE, "\nTime Statistics:", file = outputfile)
        print_with_mode(Fore.WHITE, "\tTotal Time: " + str(initialize_time + mcmc_time + z3_time), file = outputfile)
        print_with_mode(Fore.WHITE, "\tTotal Initialization and Re-initialization Time: " +  str(initialize_time), file = outputfile)
        print_with_mode(Fore.WHITE, "\tTotal SA time: " + str(mcmc_time), file = outputfile)
        print_with_mode(Fore.WHITE, "\tTotal Z3 Time: " + str(z3_time), file = outputfile)
        print_with_mode(Fore.WHITE, "\tTotal SA iterations forall threads: " + str(total_iterations), file = outputfile)
        print_with_mode(Fore.WHITE, "\tTotal Z3 calls: " + str(z3_callcount), file = outputfile)
        print_with_mode(Fore.WHITE, "\tNumber of Threads: " + str(threads) , file = outputfile)
