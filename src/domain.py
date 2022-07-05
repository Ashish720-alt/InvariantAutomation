
import numpy as np
import repr

def list_union(X, Y):
    return list(set(X + Y))

def extract_constants(P,B,T,Q):
    n = len(P[0][0]) - 2
    def extractdnfconsts(dnf):
        (const, coeff) = (set() , set([0,1]))
        for cc in dnf:
            for p in cc:
                const |= set([p[self.n+1]])
                coeff |= set(p[:-2].flatten())
        return (coeff, const)
        
    def extractptfconsts(ptf):
        (const, coeff) = (set() , set())
        for row in ptf:
                const |= set([row[self.n]])
                coeff |= set(row[:-1].flatten())  
        return (coeff, const)                 

    coeff = extractdnfconsts(self.P)[0] | extractdnfconsts(self.B)[0] | extractdnfconsts(self.Q)[0]
    const = extractdnfconsts(self.P)[1] | extractdnfconsts(self.B)[1] | extractdnfconsts(self.Q)[1]
    ptfp_list = self.T[0] + self.T[1]
    for ptfp in ptfp_list:
        coeff |= extractdnfconsts(ptfp.b)[0] | extractptfconsts(ptfp.t)[0]
        const |= extractdnfconsts(ptfp.b)[1] | extractptfconsts(ptfp.t)[1]
    return (list(coeff), list(const))


def ccl(X):
    return range(min(X), max(X+1), 1)

def gd_coeff(X):
    return list_union(X, [(-1)*x for x in X])   

def gd_const(X):
    temp = list_union(X, [(-1)*x for x in X])
    return list_union(temp, [t-1 for t in temp])

def scd(k):
    return range(-k, k+1, 1)

def npcd_const(X, r):
    ret = X
    for i in range(1, r+1):
        ret = ret + [x+i for x in X] +  [x-i for x in X]
    return list(set(ret))


def D_singlecoeff(coeff):
    return ccl(gd_coeff(list_union(scd(conf.k), coeff)))

def D_const(pc):
    return ccl(gd_const(list_union(scd(conf.k), npcd_const(pc, conf.r) )))

def D_p(P, B, T , Q):
    (coeff, const) = extract_constants(P,B,T,Q)
    pc = list_union(coeff, const)
    return (D_singlecoeff(coeff) , D_const(pc) )