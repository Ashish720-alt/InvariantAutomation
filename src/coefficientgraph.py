from itertools import product
import numpy as np
from math import acos, sqrt, gcd, pi
from functools import reduce

def removeduplicates(coeffdomain):
    i = 0
    while(1):
        vec1 = np.asarray( coeffdomain[i])
        j = i+1
        while(1): 
            if (j == len(coeffdomain)):
                break
            currvalue = coeffdomain[j]
            vec2 = np.asarray( currvalue )            
            if (acos( np.dot(vec1, vec2) / sqrt( np.dot(vec1, vec1) * np.dot(vec2, vec2) ) ) == 0):
                coeffdomain.remove(currvalue )
                j = j - 1
            j = j + 1
        i = i + 1
        if (i == len(coeffdomain)):
            break    
    return coeffdomain

def gcdnormalize(coeffdomain):
    for i,t in enumerate(coeffdomain):
        cf = reduce(gcd, list(t))
        coeffdomain[i] = tuple([int(x / cf) for x in list(coeffdomain[i])])  
    return coeffdomain

def convert_tuplecoeff_to_listcoeff(coeffdomain):
    return [ list(coeff) for coeff in coeffdomain]

def enumeratecoeffdomain(K, n):
    if (n == 1):
        return [ [-1 ], [1] ]
    singlecoeffdomain = list(range(-K, K+1, 1))
    coeffdomain = list(product( singlecoeffdomain , repeat=n))
    coeffdomain.remove( tuple([0]*n) )
    return convert_tuplecoeff_to_listcoeff(gcdnormalize(removeduplicates( coeffdomain)))


def getangle(coeff1, coeff2):
    vec1 = np.asarray( coeff1)
    vec2 = np.asarray( coeff2)
    return acos( np.dot(vec1, vec2) / sqrt( np.dot(vec1, vec1) * np.dot(vec2, vec2) ) )    

def getminangleofvector(coeff, coeffdomain):
    mintheta = pi
    for ithcoeff in coeffdomain:
        if (ithcoeff == coeff):
            continue
        mintheta = min(mintheta, getangle(coeff, ithcoeff))
    return mintheta

# Returns angle in radians
def gettheta_0(coeffdomain):
    theta_0 = 0
    for t in coeffdomain:
        theta_0 = max(theta_0, getminangleofvector(t, coeffdomain))
    return theta_0

def isneighbor(coeff1, coeff2, theta_0):
    return ( getangle(coeff1, coeff2) <= theta_0 )


def getneighborsofvector(coeff, coeffdomain, theta_0):    
    rv = []
    for i,ithcoeff in enumerate(coeffdomain):
        if (ithcoeff != coeff and isneighbor(coeff, ithcoeff, theta_0)):
            rv.append( list(coeffdomain[i]))
    return rv

# Scales only till n <= 3; for n = 4, K = 3 is allowed.
def getadjacencylist(coeffdomain, theta_0):
    n = len(coeffdomain[0])
    if (n == 1):
        return  { (1, ): [ [-1] ] , (-1, ): [ [1] ] } 
    rv = {}
    for coeff in coeffdomain:
        rv.update({ tuple(coeff) : getneighborsofvector(coeff, coeffdomain, theta_0) })
    return rv

def getrotationgraph(K, n):
    coeffdomain = enumeratecoeffdomain(K, n)
    theta_0 = gettheta_0(coeffdomain)
    # print(180*theta_0/pi)   
    return (coeffdomain, getadjacencylist( coeffdomain, theta_0))

# print(getrotationgraph(2,2)[0], '\n', getrotationgraph(2,2)[1])