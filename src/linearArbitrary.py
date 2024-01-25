
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
import sys
from sklearn.tree import DecisionTreeClassifier

PRINT_LOG = False

def listof2Darrays_to_list3D (I):
    A = [cc.tolist() for cc in I ]
    return A


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
    return ( sum( [a * b for a,b in zip( p[:-2], pt)]   )  <= p[-1]  )

def minuspointSVM(p, pt):
    return ( sum( [a * b for a,b in zip( p[:-2], pt)]   ) > p[-1]  )


def SVM(plus, minus, n):
    # Convert lists to numpy arrays

    if (len(plus) + len(minus) == 0):
        return [[ [0]*(n-2) + [-1,0]  ]] #dnfTrue
    if (len(plus) == 0):
        max_x = max(sublist[0] for sublist in minus) 
        return [[ [-1] + [0]*(n-1) + [-1, -1*max_x - 1]  ]]
    if (len(minus) == 0):
        max_x = max(sublist[0] for sublist in plus) 
        return [[ [1] + [0]*(n-1) + [-1,max_x]  ]]

    plus_actual = list(plus) #list is to create a new copy
    minus_actual = list(minus)
    
    X = np.array(plus_actual + minus_actual)
    y = np.array([1] * len(plus_actual) + [-1] * len(minus_actual))

    # Create SVM model with soft margin
    svm = SVC(kernel='linear', C=10000.0)
    svm.fit(X, y)

    # Extract coefficients of the linear classifier
    coef = svm.coef_[0]
    intercept = svm.intercept_[0]

    # Output in the specified format [coefficients, constant]
    output_gen = list(coef) + [1, -1 * intercept] #SVC learns w^Tx >= b form
    output = [-1 * x for x in output_gen]
    
    return [[output]]     

def fullSVM(plus, minus, n):
    
    # print("\t\t\tFullSVM: \n", '\t\t\t\t', '+ = ', plus, ', - = ', minus) #Debug
    
    phi = SVM(plus, minus, n) 
    
    # Normalizer required here ?!?
    
    plus_correct = [p for p in plus if pluspointSVM( phi[0][0] , p)]
    plus_wrong = [p for p in plus if not pluspointSVM( phi[0][0] , p)]
    minus_wrong = [m for m in minus if not minuspointSVM( phi[0][0] , m)]
    
    # print("\t\t\t\tClassifier = ", phi, ', +_corr = ', plus_correct, ', +_wrong = ', plus_wrong, ', -_wrong = ', minus_wrong) #Debug
    
    if ( (len(plus) > 0 and len(plus_wrong) == len(plus)) or  (len(minus) > 0 and len(minus_wrong) == len(minus)) ):
        print("SVM failed to find a classifier which correctly classifies atleast one positive and one negative point, conditioned to such a positive or negative point exists.")
        print("Failed!!")
        sys.exit()
    
    if (len(minus_wrong ) != 0):
        phi = dnfconjunction(phi, fullSVM(plus_correct, minus_wrong, n))
    if (len(plus_wrong) != 0):
        phi = dnfdisjunction(phi, fullSVM(plus_wrong, minus, n))

    return phi



def DTlearn(plus_points, minus_points, slopes):
    def getAllSlopes(slopes):
        rv = []
        n = len(slopes[0])
        for p in slopes:
            rv = rv + [p]
        for i in range(n):
            coordinate = [0]*n
            coordinate[i] = 1
            rv.append(coordinate)
        return rv
    
    def transformDataset(D, F):
        rv = []
        for d in D:
            rv.append([float(np.dot(d, f)) for f in F])

        return rv

    
    # Combine plus and minus points with labels
    Features = getAllSlopes(slopes)
    plus_labels = [1] * len(plus_points)
    minus_labels = [0] * len(minus_points)
    datapoints = transformDataset(plus_points + minus_points, Features)
    data = np.array(datapoints)
    labels = np.array(plus_labels + minus_labels)

    # Train Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(data, labels)

    tree_structure = clf.tree_

    def DTget_cc(root, leaf, leafpt):
        node = root
        rv = []
        while node != leaf and (tree_structure.children_left[node] != tree_structure.children_right[node]):
            feature_index = tree_structure.feature[node]
            threshold = tree_structure.threshold[node]
            # print(predicate, end=" -> ")
            
            if leafpt[feature_index] <= threshold:
                predicate = Features[feature_index] + [-1, threshold]
                rv.append(predicate)
                node = tree_structure.children_left[node]
            else:
                predicate = [-1*x for x in Features[feature_index]] + [-1, (-1* threshold) - 1]
                rv.append(predicate)
                node = tree_structure.children_right[node]
        return rv

    # Iterate over all leaves and print the path features
    Ls = clf.apply(datapoints[:len(plus_points)])
    DLMap = {}
    for i in range(len(Ls)):
        DLMap[Ls[i]] = datapoints[i]        
    leaves = np.where(tree_structure.children_left == tree_structure.children_right)[0]
    root_node = 0
    DT_dnf = []
    for leaf in leaves:
        if (not( leaf in DLMap.keys())):
            continue
        cc = DTget_cc(root_node, leaf, DLMap[leaf])
        DT_dnf.append(cc)

    return DT_dnf

def learnClassifier(plus, minus, n):
    SVMclassifier = fullSVM(plus, minus, n)
    # print("SVM Classifier is ", SVMclassifier)
    
    if (len(plus) == 0 or len(minus) == 0):
        return SVMclassifier
    
    coeffs = []
    for cc in SVMclassifier:
        coeffs = coeffs + [ p[:-2] for p in cc]
    
    DTclassifier = DTlearn(plus, minus, coeffs)
    # print(DTclassifier)
    
    return DTclassifier

def is_list_in_list_of_lists(main_list, sublist):
    return any(sublist == sub for sub in main_list)

def linearArbitrary(inputname, P, B, T, Q, Vars):    
    n = len(P[0][0]) - 2
    
    P_z3expr = DNF_to_z3expr( P, primed = 0)
    B_z3expr = DNF_to_z3expr(B, primed = 0)
    Q_z3expr = DNF_to_z3expr(Q, primed = 0)
    T_z3expr = genTransitionRel_to_z3expr(T)
    
    pluspoints = []
    minuspoints = []
    
    classifier = listof2Darrays_to_list3D( dnfTrue(n) )
    
    if (PRINT_LOG):
        print('t = -1 : ' , 'cex = []',', + = ', pluspoints, ', - = ', minuspoints)
        print('\t', 'ClassifierLearnt = ',  classifier , '\n\n')
    
    for t in range(0, 2000):
        (z3_correct, clause, cex) = z3_checker(P_z3expr, B_z3expr, T_z3expr, Q_z3expr, classifier)  
        
        if (z3_correct):
            print(inputname, "Success!!", classifier)
            return

        if (clause == 1):
            if (is_list_in_list_of_lists(pluspoints, cex)):
               print("Positive cex already in plus points! New positive cex = ", cex)
               print("Failed!!")
               sys.exit() 
            pluspoints = pluspoints + [cex]
            minuspoints = []            
        elif (clause == -1):
            if (is_list_in_list_of_lists(minuspoints, cex)):
               print("Minus cex already in minus points! New negative cex = ", cex)
               print("Failed!!")
               sys.exit() 
            minuspoints = minuspoints + [cex]
        else:
            hd = cex[0]
            tl = cex[0]
            if (any(hd == sublist for sublist in pluspoints)):
                if (is_list_in_list_of_lists(pluspoints, tl)):
                    print("Positive cex already in plus points! New positive cex = ", tl)
                    print("Failed!!")
                    sys.exit()                 
                pluspoints = pluspoints + [tl]
                minuspoints = []
            else:
                if (is_list_in_list_of_lists(minuspoints, hd)):
                    print("Minus cex already in minus points! New negative cex = ", hd)
                    print("Failed!!")
                    sys.exit()                 
                minuspoints = minuspoints + [hd]

        if (PRINT_LOG):
            print('t = ' + str(t) + ' :' , 'cex = ', cex, ', + = ', pluspoints, ', - = ', minuspoints)
        
        classifier = learnClassifier(pluspoints, minuspoints, n) #classifier as a 3D list  
        if (PRINT_LOG):
            print('\t', 'ClassifierLearnt = ',  classifier , '\n\n')
    
    print(inputname, "Failed!!")
    
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

