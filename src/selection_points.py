
import numpy as np
import cdd
from configure import Configure as conf
from dnfs_and_transitions import dnfconjunction, dnfnegation, dnfdisjunction, dnfTrue, dnfFalse, genLII_to_LII, transition
from math import floor, ceil
import random
from scipy.spatial import ConvexHull
from enetspointcounting import getpoints

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

# Assumes it is a convex polytope and generators is a list of lists.
def getvolume (generators, n):
    if (len(generators) < 3):
        return 0
    A = np.asarray(generators)
    return ConvexHull(A).volume

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
        def pointsatisfiescc (pt, cc):
            for p in cc:
                sat = sum(p[:-2]* pt) - p[-1]
                if (sat > 0):
                    return False
            return True
        
        point = [0]*n
        for i in range(n):
            point[i] = random.randint(minpoints[i], maxpoints[i])
        while (  not pointsatisfiescc(pt, cc) ):
            for i in range(n):
                point[i] = random.randint(minpoints[i], maxpoints[i])                    

        return point
    
    for i in range(m):
        rv.append(randomlysamplepoint(n, minpoints, maxpoints))
        
    return rv

def get_cc_pts (cc, m):
    n = len(cc[0]) - 2
    v_repr = v_representation(cc)
    vol = getvolume(v_repr, n)
    if (vol > conf.BoxesCountSmallSpace):
        return randomlysamplepointsCC(cc, m)
    else:
        return v_repr

def get_plus0 (P, e, p):
    n = len(P[0][0]) - 2
    m = getpoints(n, e, p)[0]
    P_LII_in_Dstate = dnfconjunction( P , Dstate(n), 0)    

    plus0 = []
    for cc in P_LII_in_Dstate:
        plus0 = plus0 + get_cc_pts(cc, m)

    return plus0  

def get_minus0 (Q, e, p):
    n = len(Q[0][0]) - 2
    m = getpoints(n, e, p)[0]
    negQ_LII_in_Dstate = dnfconjunction( dnfnegation(Q) , Dstate(n), 0) 

    minus0 = []
    for cc in negQ_LII_in_Dstate:
        minus0 = minus0 + get_cc_pts(cc, m)

    return minus0


def get_ICE0( T, e, p):
    n = len(T[0].b[0][0]) - 2
    m = getpoints(n, e, p)[0]
    rv = []
    def partialICE_enet (B , transitionlist, m, n):
        rv = []
        B_LII_in_Dstate = dnfconjunction( B , Dstate(n), 0)
        for cc in B_LII_in_Dstate:
            ICE_hds = get_cc_pts (cc, m)
            for hd in ICE_hds:
                for ptf in transitionlist:
                    rv.append( (hd, transition(hd, ptf)) )
        return rv

    for rtf in T:
        rv = rv + partialICE_enet (rtf.b , rtf.tlist, m, n)
    
    return rv

       


# print( randomlysamplepointsCC( np.array( [[1, 0, 1, 0] , [1, 0, -1, 2] , [0, 1, 1, 0] , [0, 1, -1, 2]]   ) , 8) )