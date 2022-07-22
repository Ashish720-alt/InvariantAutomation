from itertools import product
import numpy as np
from math import acos, sqrt, gcd, inf, pi
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

# Scales only till n <= 3; for n = 4, K = 3 is allowed.
def enumeratecoeffdomain(K, n):
    if (n == 1):
        return [ (-1, ), (1, ) ]
    singlecoeffdomain = list(range(-K, K+1, 1))
    coeffdomain = list(product( singlecoeffdomain , repeat=n))
    coeffdomain.remove( tuple([0]*n) )
    coeffdomain = gcdnormalize(removeduplicates( coeffdomain))
    # print(coeffdomain, len(coeffdomain))
    return  coeffdomain


def getangle(tuple1, tuple2):
    vec1 = np.asarray( tuple1)
    vec2 = np.asarray( tuple2)
    return acos( np.dot(vec1, vec2) / sqrt( np.dot(vec1, vec1) * np.dot(vec2, vec2) ) )    

def getminangleofvector(tuple1, coeffdomain):
    mintheta = pi
    for t in coeffdomain:
        if (t == tuple1):
            continue
        mintheta = min(mintheta, getangle(tuple1, t))
    return mintheta

def gettheta_0(coeffdomain):
    theta_0 = 0
    for t in coeffdomain:
        theta_0 = max(theta_0, getminangleofvector(t, coeffdomain))
    return theta_0

def isneighbor(tuple1, tuple2, theta_0):
    return ( getangle(tuple1, tuple2) <= theta_0 )


def getneighborsofvector(tuple1, coeffdomain, theta_0):    
    rv = []
    for i,t in enumerate(coeffdomain):
        if (t != tuple1 and isneighbor(tuple1, t, theta_0)):
            rv.append(coeffdomain[i])
    return rv

def getadjacencylist(coeffdomain):
    n = len(coeffdomain[0])
    if (n == 1):
        return [ (1, ), (-1, ) ]
    theta_0 = gettheta_0(coeffdomain)
    rv = []
    for t in coeffdomain:
        rv.append(getneighborsofvector(t, coeffdomain, theta_0))
    return rv

# enumeratecoeffdomain(5,3)
# print(isneighbor((1,2,1) , (1,1,8), 40*pi/180 ) )
# print( getneighbors( (1,2,1) , enumeratecoeffdomain(2, 3) , 40*pi/180  )  )
cd = enumeratecoeffdomain(1, 4)
# print(cd)
# theta_0 = gettheta_0(cd)
# print(180*theta_0/pi)
adjlist = getadjacencylist( cd)
for i,l in enumerate(adjlist):
    print( cd[i], '\t:\t' , l)