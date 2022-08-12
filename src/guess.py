""" Guessing a new invriant.
"""
import numpy as np
import random
from dnfs_and_transitions import deepcopy_DNF, RTI_to_LII
import copy
from configure import Configure as conf
from math import inf, sqrt
from scipy.linalg import null_space
from scipy.optimize import minimize, LinearConstraint, Bounds

def randomlysamplelistoflists(l):
    return l[np.random.choice( len(l))]


def uniformlysample_I( rotation_vertices, k1, c, d, n):
    def uniformlysample_cc(rotation_vertices, k1, n, c):
        def uniformlysample_p(rotation_vertices, k1, n):
            coeff = randomlysamplelistoflists(rotation_vertices)
            const = np.random.choice( list(range(-k1-1, k1 + 1 )) )
            return list(coeff) + [-1,const]
        return  [ uniformlysample_p(rotation_vertices, k1, n) for i in range(c)  ]
    return [ uniformlysample_cc(rotation_vertices, k1, n, c) for i in range(d)  ]

def rotationdegree(rotationneighbors):
    return len(rotationneighbors)

# C = op_norm_conj(C)
# A = np.concatenate(
#     [C[:, :self.num_var], C[:, self.num_var+1:]], axis=1)
# return float(minimize(
#     lambda x, p: np.linalg.norm(x - p),
#     np.zeros(self.num_var),
#     args=(p,),
#     constraints=[LinearConstraint(A[:, :-1], -np.inf, -A[:, -1])],
# ).fun)

def listmultiplyconstant(c, l):
    return [ c * x for x in l ]

def listadd(l1, l2):
    return [sum(p) for p in zip(l1, l2)]

# Here, posed as an ILP!
def centre_of_rotation_new(pred, newcoefficient, spin):
    # Need to convert the type of elements of this array to float?
    coeff = pred[:-2]
    const = pred[-1]
    n = len(coeff)
    sign = 1 if (spin >= 0) else -1
    v = listadd(listmultiplyconstant( 1.0/ sqrt(np.dot(newcoefficient, newcoefficient)) , newcoefficient) ,listmultiplyconstant(-1.0/ sqrt(np.dot(coeff, coeff)), coeff))
    
    # print(coeff, const, n, sign, v) #Debugging
    
    return minimize(
        lambda x, v, spin: spin * np.dot(np.array(v), np.array(x)),
        np.zeros(n), #This is the initial guess!
        args=(v,spin),
        bounds = Bounds(lb = np.full(n, conf.dspace_intmin), ub = np.full(n, conf.dspace_intmax) ),
        constraints=[LinearConstraint(np.array( [ coeff, listmultiplyconstant(-1, coeff) , newcoefficient]  ), np.array( [-np.inf, -np.inf, -100] ), np.array( [const, -const, 100] ) )], #Convert this to shape (1,n) instead of (n)?
    ).x

# print(centre_of_rotation_new( [-1,2,-1,200] , [-1,1] , 1 ))


# Returns a point (not necessarily lattice point) on the hyperplane upto some approximation error (usually 1e-9)
def centre_of_rotation(pred):
    n = len(pred) - 2
    coeff = pred[:-2]
    A = np.array([ np.array(coeff) for i in range(n)])
    ns_array = np.transpose(null_space(A))
    ns_list = [list(x) for x in ns_array] 

    def coordinate_bounds(basis):
        def coeffbounds(basisvector):
            (maxvalue, minvalue) = (max(basisvector), min(basisvector))
            (maxposvalue, minnegvalue) = (max(0.01, maxvalue), min(-0.01, minvalue) )
            U = min( conf.dspace_intmax/(maxposvalue), conf.dspace_intmin/(minnegvalue)  )
            L = max( conf.dspace_intmax/(minnegvalue), conf.dspace_intmin/(maxposvalue)  )
            return (L,U)
        return [coeffbounds(v) for v in basis]

    bounds = coordinate_bounds(ns_list)

    def samplepoint(basis, bounds, coeff, b):
        def isvalidvector(pt):
            return all([ ((val >= conf.dspace_intmin) and (val <= conf.dspace_intmax)) for val in pt ])
        n = len(basis[0])
        rv = [conf.dspace_intmax+1]*n
        K = b / np.dot(np.array(coeff), np.array(coeff))
        while( not isvalidvector(list(rv))):
            coordinates = [ np.random.uniform(I[0], I[1]) for I in bounds]
            rv = np.zeros(n, dtype = float)
            for i in range(len(basis)):
                rv = np.add(rv, coordinates[i]*np.array(basis[i]) )
            rv = np.add(rv, K*np.array(coeff))
        return list(rv)
    
    return samplepoint(ns_list, bounds, coeff, pred[-1])




def rotationtransition(oldpredicate, rotationneighbors, spin):
    # centreofrotation = centre_of_rotation(oldpredicate)
    newcoefficient = list(randomlysamplelistoflists(rotationneighbors)) 
    centreofrotation = centre_of_rotation_new(oldpredicate, newcoefficient, spin)
    const = round(np.dot(np.array(newcoefficient), np.array(centreofrotation)), 0) 
    return newcoefficient + [-1, const]


def translationtransition(predicate):
    s = np.random.choice([-1,1])
    rv = predicate.copy()
    rv[-1] = rv[-1] + s
    return rv

# call this function, then choose either rotation or translation change with probability 1/2, and then do the respective change.
# Also call the cost in the main function.
def ischange():
    return (np.random.rand() <= 1 - conf.p )

def get_index(d, c):
    return (np.random.choice( list(range(d)) ), np.random.choice( list(range(c)) ))

def isrotationchange():
    return (np.random.rand() <= conf.p_rot )




# h = [1,1,-1,3]
# a = centre_of_rotation(h)
# print(a, np.dot(np.array(h[:-2]), np.array(a)) - h[-1])