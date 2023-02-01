from itertools import product
import numpy as np
from math import acos, sqrt, gcd, pi
from functools import reduce
import pprint 
import sys
import ast
import os.path
from os import path

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

def computeRotationGraph(K, n):
    coeffdomain = enumeratecoeffdomain(K, n)
    # theta_0 = gettheta_0(coeffdomain)
    # Does large rotation angles work? We need large rotation angles
    theta_0 = pi/4
    # print(180*theta_0/pi)   
    return (coeffdomain, getadjacencylist( coeffdomain, theta_0))

def getrotationgraph(K, n):

    filename = "n" + str(n) + "K" + str(K)
    
    if (not path.isfile(filename)):
            G = computeRotationGraph(K,n)
            original_stdout = sys.stdout
            with open(filename, 'w') as f:
                sys.stdout = f # Change the standard output to the file we created.
                pprint.pprint(G[1], f)
                sys.stdout = original_stdout # Reset the standard output to its original value

    with open(filename) as f:
        data = f.read()
        E = ast.literal_eval(data)

    return ( list(E.keys()), E)

