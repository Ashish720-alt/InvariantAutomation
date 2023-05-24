
import math

def phi (n , d):
    if (d == 0):
        return 1.0
    if (d >= n):
        return 1.0 * 2**n
    
    binomk = 1
    sum = 1
    for i in range(1, d + 1):
        binomk =  ((n - i + 1) * binomk )/ i
        sum = sum + binomk    
    
    return sum    


def failprob (m , d, e):
    return 2.0 * phi(2*m, d) * (2 ** (-e * m / 2))


# vc for polytopes
# def vcrange_polytopes (n , c , d, X_c):
#     vc = d + c + X_c
#     return math.ceil(2 * (n + 1) * vc * math.log2(3 * vc))

# p is probability of success
def getpoints (d , e, p):
    # vc = d + 1 #balls
    vc = int((d**2 + 3*d )/ 2) #ellipsoids
    # print("\n", d, ": ", end = '')
    m0 = math.ceil(8 / e)
    for i in range(m0, m0 + 10000000):
        if (failprob(i, vc, e) <= 1 - p):
            return (i , failprob(i, vc, e ) )

# def findsize ():
#     for e in [0.8, 0.75, 0.7, 0.6, 0.5, 0.25, 0.125, 0.1 ]:
#         print("\n", e, ": ", end = '')
#         for d in [2, 3, 4, 5]:
#             print(getpoints(d , e, 0.9))



# findsize()