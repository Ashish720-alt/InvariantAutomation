""" Guessing a new invriant.
"""
import numpy as np
import random
from cost_funcs import f
from dnfs_and_transitions import deepcopy_DNF, RTI_to_LII
import copy
from configure import Configure as conf
from math import inf

def randomlysamplelistoflists(l):
    ind = np.random.choice( len(l))
    return l[ind]


def uniformlysampleRTI( rotation_vertices, k1, c, d, n):
    def uniformlysampleRTcc(rotation_vertices, k1, n, c):
        def uniformlysampleRTp(rotation_vertices, k1, n):
            coeff = randomlysamplelistoflists(rotation_vertices)
            # Rejection Sampling
            const = inf
            while ( const > k1 or const < -k1 ):
                pt = random.choices( list(range(conf.dspace_intmin, conf.dspace_intmax + 1)) , k = n)
                const = np.dot(np.asarray(pt), np.asarray(coeff))
            return [coeff, pt]
        
        return [ uniformlysampleRTp(rotation_vertices, k1, n) for i in range(c)  ]
    RTI = [ uniformlysampleRTcc(rotation_vertices, k1, n, c) for i in range(d)  ]
    return RTI

    # LII = RTI_to_LII(RTI)
    # (fI, costI) = f(LII, samplepoints, beta)

def translationneighbors (tp, rp, k1):
    const_old = np.dot(np.asarray(tp), np.asarray(rp))
    rv = []
    for i,value in enumerate(tp):
        tpcopy = tp.copy()
        if (value <= conf.dspace_intmax - 1 and const_old + rp[i] <= k1 and const_old + rp[i] >= -k1-1):
            tpcopy[i] = tpcopy[i] + 1
            rv.append(tpcopy)
        tpcopy = tp.copy()
        if (value >= conf.dspace_intmin + 1 and const_old - rp[i] <= k1 and const_old - rp[i] >= -k1-1):
            tpcopy[i] = tpcopy[i] - 1
            rv.append(tpcopy)
    return rv


def translationdegree(tp, rp, k1):
    return len(translationneighbors(tp, rp, k1))

def rotationdegree(rotationneighbors):
    return len(rotationneighbors)

def allowedrotations(rotationneighbors, centreofrotation, k1):
    rv = []
    
    for neighbor in rotationneighbors:
        new_const = np.dot(np.asarray(neighbor), np.asarray(centreofrotation))
        if(new_const <= k1 and new_const >= -k1-1 ):
            rv.append(neighbor)
    return rv    

def rotationtransition(rotationneighbors, centreofrotation, k1):
    allowedneighbors = allowedrotations(rotationneighbors, centreofrotation, k1)
    if (len(allowedneighbors) == 0):
        return ( [0] * len(centreofrotation) , 0)
    return ( list(randomlysamplelistoflists(allowedneighbors)) , len(allowedneighbors))


def translationtransition(trans_pred, rot_pred, k1):
    translation_neighbors = translationneighbors(trans_pred, rot_pred, k1)
    return (randomlysamplelistoflists(translation_neighbors) , len(translation_neighbors))

# call this function, then choose either rotation or translation change with probability 1/2, and then do the respective change.
# Also call the cost in the main function.
def ischange():
    return (np.random.rand() <= 1 - conf.p )

def get_index(d, c):
    return (np.random.choice( list(range(d)) ), np.random.choice( list(range(c)) ))

def isrotationchange():
    return (np.random.rand() <= conf.p_rot )




