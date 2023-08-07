
import numpy as np
import cdd
from configure import Configure as conf
from dnfs_and_transitions import dnfconjunction, dnfnegation, dnfdisjunction, dnfTrue, dnfFalse, genLII_to_LII, transition
from math import floor, ceil
import random
from scipy.spatial import ConvexHull
from enetspointcounting import getpoints
from cost_funcs import LIDNFptdistance

def Dstate(n):
    cc = np.empty(shape=(0,n+2), dtype = int)
    for i in range(n):
        p1 = np.zeros(n+2)
        p1[n] = -1
        p1[i] = -1
        p1[n+1] = -1 * conf.dspace_intmin

        p2 = np.zeros(n+2)
        p2[n] = -1
        p2[i] = 1
        p2[n+1] = conf.dspace_intmax
        
        cc = np.concatenate((cc, np.array([p1, p2], ndmin=2)))
    
    return [cc]




# Assumes cc has LI predicates only, and not genLI predicates
def v_representation (cc):
    def pred_to_matrixrow (p):
        matrixrow = np.roll(p * -1, 1)[:-1]
        matrixrow[0] = matrixrow[0] * -1
        return matrixrow

    mat = []
    for p in cc:
        mat.append(pred_to_matrixrow(p))
    
    mat_cdd = cdd.Matrix( mat, number_type='float')
    mat_cdd.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat_cdd)
    
    v_repr = poly.get_generators()
    tuple_of_tuple_generators = v_repr.__getitem__(slice(v_repr.row_size))
    list_of_list_generators = []
    for tuple_generator in tuple_of_tuple_generators:
        list_of_list_generators.append(list(tuple_generator)[1:])

    return list_of_list_generators

# def getmaxradius (generators):
#     def twoptdistance(list1, list2):
#         squares = [(p-q) ** 2 for p, q in zip(list1, list2)]
#         return sum(squares) ** .5

#     listsum = [sum(x) for x in zip(*generators)]
#     m = len(generators) * 1.0
#     if (m == 0):
#         return 0
#     centroid = [x / m for x in listsum]
    
#     distances = [ twoptdistance(centroid, x) for x in generators ]
        
#     return max (distances)

# Assumes it is a convex polytope and generators is a list of lists.
def getvolume (generators, n):
    if (len(generators) < n+1):
        return 0
    A = np.asarray(generators)
    try:
        ret = ConvexHull(A).volume
        return ret
    except:
        return 0 
        
# cc has only <=
def pointsatisfiescc (pt, cc):
    for p in cc:
        sat = sum(p[:-2]* pt) - p[-1]
        if (sat > 0):
            return False
    return True

#CC is 2d numpy array; unbounded polytopes allowed
def randomlysamplepointsCC (CC, m):
    n  = len(CC[0]) - 2
    cc = dnfconjunction([CC], Dstate(n), 0)[0]

    endpoints = v_representation(cc)
    minpoints = [ conf.dspace_intmax] * n
    maxpoints = [ conf.dspace_intmin] * n
    for pt in endpoints:
        for i in range(n):
            A = floor(pt[i])
            B = ceil(pt[i])
            if( B < minpoints[i]):
                minpoints[i] = B
            if( A > maxpoints[i]):
                maxpoints[i] = A

    rv = []
    def randomlysamplepoint (n, minpoints, maxpoints):
        
        point = [0]*n
        for i in range(n):
            point[i] = random.randint(minpoints[i], maxpoints[i] + 1)
        while (  not pointsatisfiescc(point, cc) ):
            # print("Am I stuck here?") # Sampling efficiency for affine spaces is too low.
            for i in range(n):
                point[i] = random.randint(minpoints[i], maxpoints[i] + 1)                    

        return point
    
    for i in range(m):
        rv.append(randomlysamplepoint(n, minpoints, maxpoints))
        
    return rv

def ExtremeLatticePoints( v_repr, cc, n):
    def nonintegralcoordinates( pt):
        rv = []
        for i in range(n):
            if not isinstance(pt[i], int):
                rv.append(i)
        return rv

    def binarylist(n, size):
        binaryNumber = [int(x) for x in bin(n)[2:]]
        paddedSize = size - len(binaryNumber)
        return [0] * paddedSize + binaryNumber

    rv = []
    for pt in v_repr:
        nonIntegerCoordinates = nonintegralcoordinates(pt)
        if (len(nonIntegerCoordinates) == 0):
            rv.append(pt)
        else:
            d = len(nonIntegerCoordinates)
            for i in range(1, 2**d + 1):
                pt_new = pt.copy()
                binaryList = binarylist(i, d)
                for i in range(d):
                    if (binaryList[i] == 0):
                        pt_new[nonIntegerCoordinates[i]] = floor(pt_new[nonIntegerCoordinates[i]])
                    else:
                        pt_new[nonIntegerCoordinates[i]] = ceil(pt_new[nonIntegerCoordinates[i]])    
                if (pointsatisfiescc(pt_new, cc)):
                    rv.append(pt_new)
    return rv            

def get_cc_pts (cc, m):
    n = len(cc[0]) - 2
    v_repr = v_representation(cc)
    vol = getvolume(v_repr, n)
    # r = getmaxradius (v_repr)
    if (vol > conf.SmallVolume):
        return randomlysamplepointsCC(cc, m)
    else:
        return ExtremeLatticePoints(v_repr, cc, n)

def filter_ICEtails(n, tl_list):
    rv = []
    for tl in tl_list:
        if ( LIDNFptdistance(Dstate(n), tl) == 0):
            rv.append(tl)
    return rv


#CC is 2d numpy array; unbounded polytopes allowed
def randomlysampleCC_ICEpairs (CC, m, transitions):
    n  = len(CC[0]) - 2
    cc = dnfconjunction([CC], Dstate(n), 0)[0]

    endpoints = v_representation(cc)
    minpoints = [ conf.dspace_intmax] * n
    maxpoints = [ conf.dspace_intmin] * n
    for pt in endpoints:
        for i in range(n):
            A = floor(pt[i])
            B = ceil(pt[i])
            if( B < minpoints[i]):
                minpoints[i] = B
            if( A > maxpoints[i]):
                maxpoints[i] = A


    rv = []
    
    #ICE star outputted
    def randomlysampleICEstar (n, minpoints, maxpoints, transitions):
        def pointsatisfiescc (pt, cc):
            for p in cc:
                sat = sum(p[:-2]* pt) - p[-1]
                if (sat > 0):
                    return False
            return True
        
        point = [0]*n
        for i in range(n):
            point[i] = random.randint(minpoints[i], maxpoints[i] + 1)
        

        tls = filter_ICEtails(n, [ transition(point, ptf) for ptf in transitions])
        
        while (  not pointsatisfiescc(point, cc) or tls == [] ):
            for i in range(n):
                point[i] = random.randint(minpoints[i], maxpoints[i] + 1)  
            tls = filter_ICEtails(n, [ transition(point, ptf) for ptf in transitions])                              

        return [ (point, tl) for tl in tls]
    
    for i in range(m):
        rv = rv + randomlysampleICEstar(n, minpoints, maxpoints, transitions)
        
    return rv

def get_cc_ICEheads (cc, m, transitions):
    n = len(cc[0]) - 2
    v_repr = v_representation(cc)
    vol = getvolume(v_repr, n)
    # r = getmaxradius (v_repr)
    if (vol > conf.SmallVolume):
        return randomlysampleCC_ICEpairs(cc, m, transitions)
    else:
        rv = []
        latticeVrepr = ExtremeLatticePoints(v_repr, cc, n)
        for hd in latticeVrepr:
            tls = filter_ICEtails(n, [ transition(hd, ptf) for ptf in transitions])
            if (tls == []):
                continue
            for tl in tls:
               rv.append( (hd, tl)) 
        return rv


def get_plus0 (P, m):
    n = len(P[0][0]) - 2
    P_LII_in_Dstate = dnfconjunction( P , Dstate(n), 0)    

    plus0 = []
    for cc in P_LII_in_Dstate:
        plus0 = plus0 + get_cc_pts(cc, m)

    return plus0  

def get_minus0 (Q, m):
    n = len(Q[0][0]) - 2
    negQ_LII_in_Dstate = dnfconjunction( dnfnegation(Q) , Dstate(n), 0) 

    minus0 = []
    for cc in negQ_LII_in_Dstate:
        minus0 = minus0 + get_cc_pts(cc, m)

    return minus0


def get_ICE0( T, P, Q, m):
    n = len(T[0].b[0][0]) - 2
    rv = []
    def partialICE_enet (B , P, Q, transitionlist, m, n):
        rv = []
        #The space is B 
        B_LII_in_Dstate = dnfconjunction( B , Dstate(n), 0) 
        for cc in B_LII_in_Dstate:
            rv = rv + get_cc_ICEheads(cc, m, transitionlist)
        return rv

    for rtf in T:
        rv = rv + partialICE_enet (rtf.b , P, Q, rtf.tlist, m, n)
    
    return rv

       


# print( randomlysamplepointsCC( np.array( [[1, 0, 1, 0] , [1, 0, -1, 2] , [0, 1, 1, 0] , [0, 1, -1, 2]]   ) , 8) )