# from z3 import *

from pyeda.inter import *

# u, up, x = Ints('u up x') 

# P = Lambda( [x], x < 5 )

# I = Lambda ([x], x < 3)

# C1 = Not(Implies(P[u], I[u] ))

# s = Solver()

# s.add(C1)

# r = s.check()

# print(r.__repr__())

# s.add( Implies(P(u), u == 2) )

# print(s.model().eval(x))

# I = Lambda([x], x < 5)


# print(J.body(), J.var_name[0] )

# u, up, x, xp = Ints('u up x xp') 

# I = Lambda([u], u < 3) 


# def AndLambda (P, Q):
#     return Lambda( [u],  And(P[u],Q[u]) )

# print(simplify(AndLambda(I, Lambda([u], u == 3) )[x] ) )

x, y = map(exprvar, 'xy')

print((True and False) or not (False or True))

list(iter_points([x, y]))

X = exprvars('x', 3)
f = truthtable(X, "00000001")
print(f)

print(truthtable2expr(f))