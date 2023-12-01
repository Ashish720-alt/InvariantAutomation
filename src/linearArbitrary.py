
from repr import Repr
import argparse
from input import Inputs
from selection_points import Dstate, removeduplicates, removeduplicatesICEpair, get_minus0, get_plus0
from z3verifier import DNF_to_z3expr, genTransitionRel_to_z3expr
from dnfs_and_transitions import genLII_to_LII, dnfconjunction, dnfTrue, dnfdisjunction
from z3 import *
import numpy as np
from sklearn.svm import SVC
from itertools import combinations


PRINT_LOG = True

def z3_checker(P_z3, B_z3, T_z3, Q_z3, I):
    def convert_cexlist(cexlist, ICEpair, n):
        def convert_cex(cex, ICEpair, n):
            if (ICEpair):
                return ([cex.evaluate(Int("x%s" % i), model_completion=True).as_long() for i in range(n)], [cex.evaluate(Int("x%sp" % i), model_completion=True).as_long() for i in range(n)] )
            else: 
                return [cex.evaluate(Int("x%s" % i), model_completion=True).as_long() for i in range(n)]     
        return [convert_cex(cex, ICEpair, n) for cex in cexlist]

    def __get_cex(C):
        result = []
        s = Solver()
        s.add(Not(C))
        
        while len(result) < 1 and s.check() == sat: 
  
            m = s.model()
            
            result.append(m)
            # Create a new constraint that blocks the current model
            block = []
            for d in m:
                # d is a declaration
                if d.arity() > 0:
                    raise Z3Exception("uninterpreted functions are not supported")
                # create a constant from declaration
                c = d()
                if is_array(c) or c.sort().kind() == Z3_UNINTERPRETED_SORT:
                    raise Z3Exception("arrays and uninterpreted sorts are not supported")
                block.append(c != m[d])    
     
            s.add(Or(block))
        
        else:
            if len(result) < 1 and s.check() != unsat: 
                print("Solver can't verify or disprove")
                return result
        return result

    #P -> I
    def __get_cex_plus(P_z3, I, n):
        I_z3 = DNF_to_z3expr( I, primed = 0)
        plus = convert_cexlist(__get_cex(Implies(P_z3, I_z3)), 0, n)
        if (len(plus) > 0):
            return plus
        return []

    #B & I & T => I'
    def __get_cex_ICE(B_z3, I, T_z3, n):
        def __get_cex_ICE_givenI(B_z3, T_z3, n, I_z3, Ip_z3):
            return convert_cexlist(__get_cex(Implies(And(B_z3, I_z3, T_z3), Ip_z3)), 1, n) 

        I_z3 = DNF_to_z3expr( I, primed = 0)
        Ip_z3 = DNF_to_z3expr(I, primed = 1)                  
        rv = __get_cex_ICE_givenI(B_z3, T_z3, n, I_z3, Ip_z3)
        if (rv != []):
            return rv
        
        return []            


    # I -> Q
    def __get_cex_minus(I, Q_z3, n):
        I_z3 = DNF_to_z3expr( I, primed = 0)
        minus = convert_cexlist(__get_cex(Implies(I_z3, Q_z3)), 0, n) 
        if (len(minus) > 0):
            return minus
        return []
    
    n = len(I[0][0]) - 2 
    cex_plus = __get_cex_plus(P_z3, I, n)
    if len(cex_plus) != 0:
        return (0, 1, cex_plus[0])
    
    cex_minus = __get_cex_minus(I, Q_z3, n)
    if len(cex_minus) != 0:
        return (0, -1, cex_minus[0])
    
    cex_ICE = __get_cex_ICE(B_z3, I, T_z3, n)
    if len(cex_ICE) != 0:
        return (0, 0, cex_ICE[0])
    
    return (1, None, [])

def pluspointSVM(p, pt):
    return ( sum( [a * b for a,b in zip( p[:-2], pt)]   )  >= p[-1]  )

def minuspointSVM(p, pt):
    return ( sum( [a * b for a,b in zip( p[:-2], pt)]   ) < p[-1]  )



def SVM(plus, minus, baseplus, baseminus):
    # Convert lists to numpy arrays
    
    

    n = len(baseplus)
    if (len(plus) + len(minus) == 0):
        return [[ [0]*(n-2) + [-1,0]  ]] #dnfTrue
    
    plus_actual = list(plus) #list is to create a new copy
    minus_actual = list(minus)
    
    if (len(plus) == 0):
        plus_actual = plus_actual + [baseplus]
    if len(minus) == 0:
        minus_actual = minus_actual = [baseminus]
    
    # print("SVM", plus_actual, minus_actual, end = '') #Debug
    
    X = np.array(plus_actual + minus_actual)
    y = np.array([1] * len(plus_actual) + [-1] * len(minus_actual))

    # Create SVM model with soft margin
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X, y)

    # Extract coefficients of the linear classifier
    coef = svm.coef_[0]
    intercept = svm.intercept_[0]

    # Output in the specified format [coefficients, constant]
    output = list(coef) + [1, intercept] #SVC learns w^Tx >= b form
    
    # print( output) #Debug
    
    return [[output]]     

def fullSVM(plus, minus, baseplus, baseminus):
    
    # print("FullSVM", plus, minus, baseplus, baseminus) #Debug
    
    phi = SVM(plus, minus, baseplus, baseminus) 
    plus_correct = [p for p in plus if pluspointSVM( phi[0][0] , p)]
    plus_wrong = [p for p in plus if not pluspointSVM( phi[0][0] , p)]
    minus_wrong = [m for m in minus if not minuspointSVM( phi[0][0] , m)]
    
    # print("classifier", phi, plus_correct, plus_wrong, minus_wrong) #Debug
    
    if (len(minus_wrong ) != 0):
        phi = dnfconjunction(phi, fullSVM(plus_correct, minus_wrong, baseplus, baseminus))
    if (len(plus_wrong) != 0):
        phi = dnfdisjunction(phi, fullSVM(plus_wrong, minus, baseplus, baseminus))

    return phi

def learnClassifier(plus, minus, baseplus, baseminus):
    SVMclassifier = fullSVM(plus, minus, baseplus, baseminus)
    # coeff = []
    # for cc in SVMclassifier:
    #     coeff = coeff + [ p[:-2] for p in cc]
    
    return SVMclassifier



def linearArbitrary(inputname, P, B, T, Q, Vars):    
    n = len(P[0][0]) - 2
    
    P_z3expr = DNF_to_z3expr( dnfconjunction(P, Dstate(n), 1), primed = 0)
    B_z3expr = DNF_to_z3expr(dnfconjunction(B, Dstate(n), 1), primed = 0)
    Q_z3expr = DNF_to_z3expr(dnfconjunction(Q, Dstate(n), 1), primed = 0)
    T_z3expr = genTransitionRel_to_z3expr(T)
    
    basepluspoint = get_plus0(P, 1)[0]
    baseminuspoint = get_minus0(Q, 1)[0]
    
    pluspoints = []
    minuspoints = []
    
    classifier = dnfTrue(n)
    
    if (PRINT_LOG):
        print(pluspoints, minuspoints, classifier , '\n')
    
    for _ in range(0, 2000):
        
        
        LII = [ np.array([[-x for x in p] for p in cc]) for cc in classifier  ]  
        
        (z3_correct, clause, cex) = z3_checker(P_z3expr, B_z3expr, T_z3expr, Q_z3expr, LII)  
        
        if (z3_correct):
            print(inputname, "YES", classifier)
            return

        if (clause == 1):
            pluspoints = pluspoints + [cex]
            minuspoints = []            
        elif (clause == -1):
            minuspoints = minuspoints + [cex]
        else:
            hd = cex[0]
            tl = cex[0]
            if (any(hd == sublist for sublist in pluspoints)):
                pluspoints = pluspoints + [tl]
                minuspoints = []
            else:
                minuspoints = minuspoints + [hd]

        if (PRINT_LOG):
            print(cex, pluspoints, minuspoints, end = '')
        
        classifier = learnClassifier(pluspoints, minuspoints, basepluspoint, baseminuspoint) #classifier as a 3D list  
        if (PRINT_LOG):
            print('  ', classifier , '\n')
    
    print(inputname, "NO")
    
    return
    
    


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
                obj = getattr(getattr(Inputs, subfolder), inp)
                name = subfolder + "." + inp
                linearArbitrary(name, obj.P, obj.B , obj.T , obj.Q, obj.Var)
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
                        obj = getattr(getattr(Inputs, subfolder), inp)
                        linearArbitrary(first_name + "." + last_name, obj.P, obj.B , obj.T , obj.Q, obj.Var)


# #Decision Trees part
# from itertools import combinations

# def generate_predicates(coefficients):
#     n = len(coefficients[0])
#     predicates = []
#     for coef in coefficients:
#         for i in range(n):
#             for comb in combinations(range(n), i + 1):
#                 predicate = [0] * (2 * n)
#                 for idx in comb:
#                     predicate[idx] = coef[idx]
#                 predicate[-1] = coef[-1]
#                 predicates.append(tuple(predicate))
#     return predicates

# def format_predicate(predicate):
#     n = len(predicate) // 2
#     terms = []
#     for i in range(n):
#         coef1 = predicate[i]
#         coef2 = predicate[i + n]
#         if coef1 != 0:
#             if coef2 >= 0:
#                 terms.append(f"{coef1}x{i} + {coef2}")
#             else:
#                 terms.append(f"{coef1}x{i} - {abs(coef2)}")
#     terms.append(f" <= {predicate[-1]}")
#     return ' '.join(terms)

# def build_classifier(plus_points, minus_points, coefficients):
#     predicates = generate_predicates(coefficients)
#     X = plus_points + minus_points
#     y = [1] * len(plus_points) + [0] * len(minus_points)

#     for predicate in predicates:
#         indices = [i for i in range(len(predicate) - 1) if predicate[i] != 0]
#         X_transformed = [[point[i] for i in indices] for point in X]

#         from sklearn.tree import DecisionTreeClassifier
#         clf = DecisionTreeClassifier()
#         clf.fit(X_transformed, y)
#         if clf.score(X_transformed, y) == 1:
#             formatted_predicate = format_predicate(predicate)
#             return f"Formula in DNF form: {formatted_predicate}"
    
#     return "Error: Unable to classify using provided coefficients"

# # Example usage:
# plus_points = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
# minus_points = [[-1, -2, -3], [-2, -3, -4], [-3, -4, -5]]
# coefficients = [[2, 3, 1], [-2, -3, -1], [1, 1, 1, 0]]

# result = build_classifier(plus_points, minus_points, coefficients)
# print(result)