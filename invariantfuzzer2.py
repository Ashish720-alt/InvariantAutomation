from z3 import *


x, xp = Ints('x xp') 

P_given = x == 0
B_given = x < 5
T_given = xp == x + 1
Q_given = x == 5

K = 10

# Returns true or a counterexample
def Check(mkConstraints, I_i, I_f, P , B, T , Q):
    s = Solver()
    # Add the negation of the conjunction of constraints
    s.add(Not(mkConstraints(I_i, I_f, P , B, T , Q)))
    r = s.check()
    output = r.__repr__()
    if output == "sat":
        #print("sat")
        return s.model()
    elif output == "unsat":
        #print("unsat")
        return
    else:
        print("Solver can't verify or disprove, it says: %s for invariant %s" %(r, I_i))

#Returns the conjunction of the CHC clauses of the system 

#Need to change this!!
def System(I_i, I_f, P , B, T , Q):
    # P(x) -> I(x)
    c1 = Implies(P , I_i )
    # P(x) /\ B(x) /\ T(x,xp) -> I(xp) 
    c2 = Implies(And(B , I_i , T ) , I_f ) 
    # I(x) /\ ~B(x) -> Q(x)
    c3 = Implies(And(I_i , Not(B )), Q ) 
    return And(c1, c2, c3)


cex_List = []
# Correct invariant is x <= 5
I_guess_i = x < 3
I_guess_f = xp < 3 # replace all occurrences of x in I_guess_i by xp

for i in range(K):
    cex = Check(System, I_guess_i, I_guess_f, P_given, B_given, T_given, Q_given)
    if cex is None:
        break
    # This is actual code, which gives same counterexamples after 3 different ones. (Actually after these it, there are no more - the issue is that Z3 still considers the system solvable?)
    if( cex.evaluate(I_guess_i(x)) ):
        I_guess_i = Or( I_guess_i, x = cex.evaluate(xp) )
        #I_guess_i = Lambda ( [t], Or( I_guess.body(t), t == cex.evaluate(xp) ) )  # Check this update procedure!!

    else:
        I_guess_i = Or( I_guess_i, x = cex.evaluate(xp) )
        #I_guess_i = Lambda([t], Or( I_guess.body(t), t == cex.evaluate(x) ) )


    cex_List.append(cex)


# Print the list of counterexamples.
print(cex_List)