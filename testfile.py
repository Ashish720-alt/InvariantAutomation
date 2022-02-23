from z3 import *


u, up, x = Ints('u up x') 

# P = (x < 5)

# I = (x < 3)

# C1 = Not(Implies(P, I ))

# print(C1)

# s = Solver()

# s.add(C1)

# r = s.check()

# print(s.model().eval(x))

# print(r.__repr__())

I = Lambda([x], x < 5)



print(J.body(), J.var_name[0] )