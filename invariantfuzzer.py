from z3 import *


u, up, x, xp = Ints('u up x xp') 

P_given = Lambda([u], u == 0)
B_given = Lambda([u], u < 5)
T_given = Lambda([u], Lambda ([up] , up == u + 1 ))
Q_given = Lambda([u], u == 5)

K = 10

# Returns true or a counterexample
def Check(mkConstraints, I, P , B, T , Q):
    s = Solver()
    # Add the negation of the conjunction of constraints
    s.add(Not(mkConstraints(I, P , B, T , Q)))
    r = s.check()
    output = r.__repr__()
    if output == "sat":
        #print("sat")
        return s.model()
    elif output == "unsat":
        #print("unsat")
        return
    else:
        print("Solver can't verify or disprove, it says: %s for invariant %s" %(r, I))

#Returns the conjunction of the CHC clauses of the system 
def System(I, P , B, T , Q):
    # P(x) -> I(x)
    c1 = Implies(P[x], I[x])
    # P(x) /\ B(x) /\ T(x,xp) -> I(xp) 
    c2 = Implies( And(B[x], I[x], T[x][xp] ) , I[xp]) 
    # I(x) /\ ~B(x) -> Q(x)
    c3 = Implies( And(I[x], Not(B[x]) ) , Q[x]) 
    return And(c1, c2, c3)


cex_List = []

# Correct invariant is x <= 5
I_guess = Lambda([u], u < 3) 

for i in range(K):
    cex = Check(System, I_guess, P_given, B_given, T_given, Q_given)
    if cex is None:
        break
    # This is actual code, which gives same counterexamples after 3 different ones. (Actually after these it, there are no more - the issue is that Z3 still considers the system solvable?)
    if( cex.evaluate(I_guess[x]) ):
        I_guess = simplify(Lambda ([u], Or( I_guess[u], u == cex.evaluate(xp) ) ))  
    else:
        I_guess = simplify(Lambda([u], Or( I_guess[u], u == cex.evaluate(x) ) ))

    cex_List.append(cex)

# Prints the final invariant.
# print(simplify(I_guess[x]))

# Print the list of counterexamples.
print(cex_List)


