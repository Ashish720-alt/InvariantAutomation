from z3 import *


x, xp = Ints('x xp') 

P = lambda x: x == 0
B = lambda x: x < 5
T = lambda x, xp: xp == x + 1
Q = lambda x: x == 5 

s = 10

# Returns true or a counterexample
def Check(mkConstraints, I, P , B, T , Q):
    s = Solver()
    # Add the negation of the conjunction of constraints
    s.add(Not(mkConstraints(I, P , B, T , Q)))
    r = s.check()
    if r == sat:
        return s.model()
    elif r == unsat:
        return {}
    else:
        print("Solver can't verify or disprove, it says: %s for invariant %s" %(r, I))

#Returns the conjunction of the CHC clauses of the system 
def System(I, P , B, T , Q):
    # P(x) -> I(x)
    c1 = Implies(P(x), I(x))
    # P(x) /\ B(x) /\ T(x,xp) -> I(xp) 
    c2 = Implies(And(B(x), I(x), T(x, xp)) , I(xp))
    # I(x) /\ ~B(x) -> Q(x)
    c3 = Implies(And(I(x), Not(B(x))), Q(x))
    return And(c1, c2, c3)


cex_List = []
I_guess = lambda x: x < 3

for i in range(s):
    cex = Check(System, I_guess, P, B, T, Q)
    if cex == {}:
        break
    
    # This is an approximation to the correct code, it runs; but for it only gives 3 distinct counterexamples; after that it just starts repeating the counterexamples, why?
    I_guess = lambda t, old_I_guess=I_guess: Or(old_I_guess(t), t == cex[xp] , t == cex[x])

    # This is actual code, which doesn't even run.

    # if I_guess(cex[x]):
    #     print(cex[x])
    #     I_guess = lambda t, old_I_guess=I_guess: Or(old_I_guess(t), t == cex[xp] )
    # else:
    #     I_guess = lambda t, old_I_guess=I_guess: Or(old_I_guess(t), t == cex[x])
    cex_List.append(cex)


# Print the list of counterexamples.
print(cex_List)

