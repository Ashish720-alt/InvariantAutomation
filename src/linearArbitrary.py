
import sys
from configure import Configure as conf
from cost_funcs import cost
from guess import initialInvariant, rotationtransition, translationtransition, get_index, isrotationchange, k1list, SAconstantlist, getNewRotConstant, getNewTranslationConstant, experimentalSAconstantlist
from repr import Repr
from numpy import random
from z3verifier import z3_verifier
from print import initialized, statistics, z3statistics, invariantfound, timestatistics, prettyprint_samplepoints, noInvariantFound
from print import SAexit, SAsuccess, samplepoints_debugger, SAfail
from dnfs_and_transitions import  list3D_to_listof2Darrays, dnfconjunction, dnfnegation
from timeit import default_timer as timer
from math import log, floor
import argparse
from input import Inputs, input_to_repr
import multiprocessing as mp
from invariantspaceplotter import plotinvariantspace
from selection_points import removeduplicates, removeduplicatesICEpair, get_longICEpairs

def linearArbitrary(inputname, repr: Repr):
    print(repr.P , repr.Q , repr.T , repr.B)
    return
    
    

print("huh")
parser = argparse.ArgumentParser(description='LinearArbitraryDescription')
parser.add_argument('-i', '--input', type=str, help='Input object name')
parser.add_argument('-a', '--all', action='store_true', help='Run all inputs')
parse_res = vars(parser.parse_args())
if parse_res['all']:
    if (parse_res['input'] is not None):
        print(parser.print_help())
        print("Please specify either input object name or all inputs")
        exit(1)
    for subfolder in dir(Inputs):
        for inp in dir(getattr(Inputs, subfolder)):
            try: 
                if inp.startswith("__") or subfolder.startswith("__"):
                    continue
                print("Running input " + subfolder + "." + inp)
                repr = input_to_repr(getattr(getattr(Inputs, subfolder), inp), None, None)
                name = subfolder + "." + inp
                linearArbitrary(name, repr)
            except Exception as e:
                print(f"Error {e} in input " + subfolder + "." + inp)
            print("----------------------")
else:
    if parse_res['input'] is None:
        print(parser.print_help())
        print("Please specify input object name")
        exit(1)
    else:
        (first_name, last_name) = parse_res['input'].split('.')
        for subfolder in Inputs.__dict__:
            if subfolder == first_name:
                for inp in getattr(Inputs, subfolder).__dict__:
                    if inp == last_name:
                        linearArbitrary(first_name + "." + last_name, input_to_repr(getattr(getattr(Inputs, subfolder), inp), parse_res['c'], parse_res['d']))
