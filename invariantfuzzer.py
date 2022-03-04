from z3 import *
import numpy as np 

# Although same code as *_lambda.py, this always works (never repeats cex)
# For I_g = x > 2, C3's cexlist repeats cex; only gets x = 6 and x = 7. But actually x = 6,7,8,9,10,11, .. infinity are all cex.

s = 10

# Only for 1 variable case
x, xp = Ints('x xp')

P = lambda x: x == 0
B = lambda x: x < 5
T = lambda x, xp: xp == x + 1
Q = lambda x: x == 5

# Correct invariant is x <= 5
I_g = lambda x: x > 2

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
            I_u = lambda x: Or(predicate, x == cex.evaluate(x))
        else:
            I_u = lambda t, old_I_u=I_u: Or(old_I_u(t), t == cex.evaluate(x))
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
            I_u = lambda x: Or(predicate, x == cex.evaluate(x), x == cex.evaluate(xp))
        else:
            I_u = lambda t, old_I_u=I_u: Or(old_I_u(t), t == cex.evaluate(xp), t == cex.evaluate(x))
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
            I_u = lambda x: And(predicate, x != cex.evaluate(x))
        else:
            I_u = lambda t, old_I_u=I_u: And(old_I_u(t), t != cex.evaluate(x))
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


'''Operator code:
g = 2
ge = 1
eq = 0
le = -1
l = -2 '''

P1 = np.array([1,0,0], ndmin = 3)

def distance_point_hyperplane ( p, L):
    x = float(p[0])
    L_endpoint_1 = float(L[2]) / float(L[0]) # For 1D only; for > 1 D, compute both endpoints, and distance is minimum of distance from these endpoints and also check distance of line and whether this distance lies within segment
    d = x - L_endpoint_1
    if (L[1] == 0):
        return abs(d)
    if ((L[1] > 0 and L[0]*d > 0) or (L[1] < 0 and L[0]*d < 0)):  # L[0]*d > 0 is short for (L[0]>0 and d > 0 ) or (L[0] < 0 and d < 0)
        d = 0
    return abs(d)

def distance_point_conjunctiveClause (p , C):
    d = float('inf')
    for L in C:
        d = min(d, distance_point_hyperplane(p, L))
    return d

def distance_point_DNF(p, D):
    d = float('inf')
    for C in D:
        d = min(d, distance_point_conjunctiveClause(p, C))
    return d

#Testing:
print(distance_point_hyperplane( np.array( [1], ndmin = 1), np.array([2,-2,3] , ndmin = 1)) )

print(distance_point_conjunctiveClause( np.array( [-3], ndmin = 1), np.array( [ [1,1,10], [1,0,0] ] , ndmin = 2)) ) 

print(distance_point_DNF( np.array( [-3], ndmin = 1), np.array( [ [ [-7,-2,3], [1,0,2], [3, 1, 4] ], [ [1,1,10], [1,0,0] ] ] , dtype=object   )) )

def distance_point_predicate(z, P):
    predicate = P(x)
    S = np.array([], ndmin = 3)
    # Convert P(x) into np.array form and store in S. <- This is hard!
    # compute distance of point z from P <- This is implemented by function distance_point_DNF.




