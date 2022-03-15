from z3 import *

import numpy as np


'''

# Replaced the counterexample code with python's 'and' and 'or' instead of Z3's 'and' and 'or' and it gives error.


s = 10

#I_g  


# Only for 1 variable case
x, xp = Ints('x xp')

P = lambda x: x == 0
B = lambda x: x < 5
T = lambda x, xp: xp == x + 1
Q = lambda x: x == 5

# Correct invariant is x <= 5
I_g = lambda x: x > 2 # [[[1 2 2]]]

def C1(I):
    return Implies(P(x), I(x))

def C2(I):
    return Implies(And(B(x), I(x), T(x, xp)) , I(xp))

def C3(I):
    return Implies(And(I(x), Not(B(x))), Q(x))

def System(I):
    return And(C1(I), C2(I), C3(I))

# Returns true or a counterexample
def Check(C, I):
    s = Solver()
    # Add the negation of the conjunction of constraints
    s.add(Not(C))
    r = s.check()
    output = r.__repr__()
    if output == "sat":
        return s.model()
    elif output == "unsat":
        return None
    else:
        print("Solver can't verify or disprove, it says: %s for invariant %s" %(r, I))
        return None

def GenerateCexList_C1 (I):
    cex_List = []
    I_u = I
    predicate = I_u(x)
    for i in range(s):
        cex = Check(C1(I_u), I_u)
        if cex is None:
            break
        if i == 0:
            I_u = lambda x: predicate or x == cex.evaluate(x)
        else:
            I_u = lambda t, old_I_u=I_u: old_I_u(t) or t == cex.evaluate(x)
        cex_List.append(cex)

    # Print the list of counterexamples.
    # print(cex_List)

    return cex_List

# There are two options to search for here - In fact if we want cex for original invariant, then we need to take the And option, no; otherwise we get a cex chain, no?
def GenerateCexList_C2 (I):
    cex_List = []
    I_u = I
    predicate = I_u(x)
    for i in range(s):
        cex = Check(C2(I_u), I_u)
        if cex is None:
            break
        if i == 0:
            I_u = lambda x: predicate or x == cex.evaluate(x) or x == cex.evaluate(xp)
        else:
            I_u = lambda t, old_I_u=I_u: old_I_u(t) or t == cex.evaluate(xp) or t == cex.evaluate(x)
        cex_List.append(cex)

    # Print the list of counterexamples.
    # print(cex_List)

    return cex_List

def GenerateCexList_C3 (I):
    cex_List = []
    I_u = I
    predicate = I_u(x)
    for i in range(s):
        cex = Check(C3(I_u), I_u)
        if cex is None:
            break
        if i == 0:
            # I_u = lambda x: And(predicate, x != cex.evaluate(x))
            I_u = lambda x: predicate and (x > cex.evaluate(x) or x < cex.evaluate(x)) 
        if i == 1:
            I_u = lambda t, old_I_u=I_u: old_I_u(t) and (x > cex.evaluate(x) or x < cex.evaluate(x))
        else:
            I_u = lambda t, old_I_u=I_u: (old_I_u(t) and t != cex.evaluate(x))
        cex_List.append(cex)

    # Print the list of counterexamples.
    # print(cex_List)

    return cex_List

# Get cexList for each Clause.s
C1_cexList = GenerateCexList_C1 (I_g)
print(C1_cexList)


C2_cexList = GenerateCexList_C2 ( I_g)
print(C2_cexList)

C3_cexList = GenerateCexList_C3 ( I_g)
print(C3_cexList)

'''


# for 1D case only
def convert_predicate_to_lambda ( P  ):
    if (P[1] == 0):
        return lambda x : P[0] * x == P[2]
    elif (P[1] == -1):
        return lambda x : P[0] * x <= P[2]
    elif (P[1] == -2):
        return lambda x : P[0] * x < P[2]
    elif (P[1] == 1):
        return lambda x : P[0] * x >= P[2]
    elif (P[1] == 2):
        return lambda x : P[0] * x > P[2]

# Testing the function:
# S = convert_predicate_to_lambda(np.array([1,0,0], ndmin = 1) )
# print(S(-1), S(-2) , S(0) , S(1) , S(2))


''' This function works for only python's and and gives error with Z3's And (so this version gives error) '''
# for 1D case only
def convert_conjunctiveClause_to_lambda ( C  ):
    partial_converted = lambda x: True
    for P in C:
        predicate = convert_predicate_to_lambda( P)
        partial_converted = lambda x, old_partial_converted=partial_converted: And(old_partial_converted(x), predicate(x) ) 
    return partial_converted

# Testing the function:
P1 = np.array( [[1,0,0] ], ndmin = 2)
S2 = convert_conjunctiveClause_to_lambda( P1 )
print(S2(-1), S2(-2) , S2(0) , S2(1) , S2(2))