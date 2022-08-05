""" Guessing a new invriant.
"""
import numpy as np
import random
from dnfs_and_transitions import deepcopy_DNF, RTI_to_LII
import copy
from configure import Configure as conf
from math import inf
from scipy.linalg import null_space

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




def rotationtransition(oldpredicate, rotationneighbors):
    centreofrotation = centre_of_rotation(oldpredicate)
    newcoefficient = list(randomlysamplelistoflists(rotationneighbors)) 
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