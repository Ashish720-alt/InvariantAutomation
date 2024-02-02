
import numpy as np
import cdd
from configure import Configure as conf
from dnfs_and_transitions import dnfconjunction, dnfnegation, dnfdisjunction, dnfTrue, dnfFalse, genLII_to_LII, transition
from math import floor, ceil, inf
import random
from scipy.spatial import ConvexHull
from enetspointcounting import getpoints
from cost_funcs import LIDNFptdistance
from sympy import * 
import itertools
import copy
import scipy

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



# def LatticeApproximatePoint( pt, cc):
#     n = len(pt)
#     def nonintegralcoordinates( pt):
#         rv = []
#         for i in range(n):
#             if not isinstance(pt[i], int):
#                 rv.append(i)
#         return rv

#     def binarylist(n, size):
#         binaryNumber = [int(x) for x in bin(n)[2:]]
#         paddedSize = size - len(binaryNumber)
#         return [0] * paddedSize + binaryNumber

#     nonIntegerCoordinates = nonintegralcoordinates(pt)
#     if (len(nonIntegerCoordinates) == 0):
#         return pt
#     else:
#         d = len(nonIntegerCoordinates)
#         for i in range(1, 2**d + 1):
#             pt_new = pt.copy()
#             binaryList = binarylist(i, d)
#             for i in range(d):
#                 if (binaryList[i] == 0):
#                     pt_new[nonIntegerCoordinates[i]] = floor(pt_new[nonIntegerCoordinates[i]])
#                 else:
#                     pt_new[nonIntegerCoordinates[i]] = ceil(pt_new[nonIntegerCoordinates[i]])    
#             if (pointsatisfiescc(pt_new, cc)):
#                 return pt_new
   
#     return []      


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
        print("Returning 0 Volume case 1")
        return 0
    A = np.asarray(generators)
    try:
        ret = ConvexHull(A).volume
        print("Returning NonZero Volume" + str(ret))
        return ret
    except:
        print("Returning 0 Volume case 2")
        return 0 
        
# cc has only <=
def pointsatisfiescc (pt, cc):
    for p in cc:
        sat = sum(p[:-2]* pt) - p[-1]
        if (sat > 0):
            return False
    return True

# # Affine space is AX = B, and other inequalities are given by cc ineq
def linearDiophantineSolution(A, b_t):
    def unimodularrowreduction(A, E):
        def step3(A, i, j, k):
            A[i] = A[i] - (A[j] * k)
            return   
        
        def swaprows(A, i, j):
            temp = A[i].copy()
            A[i] = A[j]
            A[j] = temp
            return
            
        def changeSign(A, i):
            A[j] = A[j] * -1
            
        (rows,columns) = A.shape
        # print(A)
        leadingcoeffIndex = 0
        for t in range(columns):
            while (1):
                j = -1
                curr_val = inf
                for i in range(leadingcoeffIndex, rows):
                    if (A[i][t] == 0):
                        continue
                    if (abs(A[i][t]) < curr_val):
                        curr_val = abs(A[i][t])
                        j = i
                if (j == -1):
                    break

                flag = 0
                for i in range(leadingcoeffIndex, rows):
                    if (i == j or A[i][t] == 0):
                        continue
                    flag = 1
                    k = floor(abs(A[i][t]) / abs(A[j][t]))
                    if (A[i][t] * A[j][t] < 0):
                        k = -k
                    step3(A, i, j, k)
                    step3(E, i, j, k)
                    # print(A)
                if (flag == 0):
                    swaprows(A, j, leadingcoeffIndex)
                    swaprows(E, j, leadingcoeffIndex)
                    if (A[leadingcoeffIndex][t] < 0):
                        changeSign(A, leadingcoeffIndex)
                        changeSign(E, leadingcoeffIndex)
                    leadingcoeffIndex = leadingcoeffIndex + 1
                    # print(A)
                    break 
                
                
        return (A, E)

    (A_row, A_column) = A.shape
    (R, T) = unimodularrowreduction( np.transpose(A), np.identity(A_column, dtype = int))
    R_t = np.transpose(R)
    knownVars = []
    for i in range(A_row):
        col = R_t[i]    
        nonZeroEntryCount = len(col)
        for j in range(len(knownVars), len(col)): #Not first zero element of column, but first zero element starting from index for new element.
            if (col[j] == 0):
                nonZeroEntryCount = j 
                break
        
        if (nonZeroEntryCount > len(knownVars)): #Get Value of a variable
            Nr = b_t[i] - int(np.dot(np.array(knownVars), col[:nonZeroEntryCount - 1]))
            Dr = int(col[nonZeroEntryCount - 1])
            if (Nr % Dr == 0):
                knownVars.append( Nr // Dr  )
            else:
                # print("No integer solution to b^T = k^TR")
                return ([], [])
        else:  #Check Stage
            if (int(np.dot(np.array(knownVars), col[:nonZeroEntryCount])) == b_t[i]):
                continue
            else:
                # print("No solution to b^T = k^TR")
                return ([], [])
    

    
    knownValue = [0] * (A_column)
    for j in range(len(knownVars)):
        knownValue = knownValue + knownVars[j] * T[j]
    colvectors = []
    for j in range(len(knownVars), A_column):
        colvectors.append(T[j].tolist())
    
    return (knownValue, colvectors)

def isEmpty(cc):
    v_repr = v_representation(cc)
    return (len(v_repr) == 0)

def isAffine(cc):
    p_set = set()
    for p in cc:
        p_tuple = tuple(p)
        negation = [-x for x in p]
        negation[-2] = -1
        if (negation == p.tolist()):
            continue
        if tuple(negation) in p_set:
            return True
        else:
            p_set.add(p_tuple)
    return False

def getAffineNonaffinepreds(cc):
    p_set = set()
    affinepreds = []
    affinepreds_LII = []
    for p in cc:
        p_tuple = tuple(p)
        negation = [-x for x in p]
        if tuple(negation) in p_set:
            affinepreds.append( negation[:-2] + [0] + [negation[-1]] )
            affinepreds_LII.append(negation[:-2] + [-1] + [negation[-1]]) 
            affinepreds_LII.append(p[:-2] + [-1] + [p[-1]]) 
        else:
            p_set.add(p_tuple)
    
    NonAffinepreds = [p for p in cc if p not in affinepreds_LII ]
    
    return False


def getAffine(cc):
    n = len(cc[0]) - 2
    ccList = cc.tolist()
    # affinepreds = []
    # NonAffinepreds = []
    # for p in ccList:
    #     if (p[-2] == 0):
    #         affinepreds.append(p)
    #     else:
    #         NonAffinepreds.append(p)

    p_set = set()
    affinepreds = []
    affinepreds_LII = []
    for p in ccList:
        p_tuple = tuple(p)
        negation = [-x for x in p]
        negation[-2] = -1
        if tuple(negation) in p_set:
            affinepreds.append( negation[:-2] + [0] + [negation[-1]] )
            affinepreds_LII.append(negation[:-2] + [-1] + [negation[-1]]) 
            affinepreds_LII.append(p[:-2] + [-1] + [p[-1]]) 
        else:
            p_set.add(p_tuple)
    
    NonAffinepreds = [p for p in ccList if p not in affinepreds_LII ]

    if NonAffinepreds == []:
        NonAffinepreds = Dstate(n)[0].tolist()
    else:
        NonAffinepreds = dnfconjunction([np.array(NonAffinepreds, ndmin = 2)], Dstate(n) , 0)[0].tolist()
    
    A = []
    b = []
    for p in affinepreds:
        A.append(p[:-2])
        b.append(p[-1])
    nonA = []
    nonb = []
    for p in NonAffinepreds:
        nonA.append(p[:-2])
        nonb.append(p[-1])
    return (np.array(A, ndmin = 2, dtype = int), np.array(b, ndmin = 1, dtype = int), np.array(nonA, ndmin = 2, dtype = int), np.array(nonb, ndmin = 1, dtype = int))
        
            


#CC is 2d numpy array; unbounded polytopes allowed
def randomlysamplepointsCC (cc, m):
    def randomlysamplepoints(endpoints, cc):
        n = len(endpoints[0])
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

        points = []
        for i in range(m):
            point = [0]*n
            for i in range(n):
                point[i] = random.randint(minpoints[i], maxpoints[i] + 1)
            while (  not pointsatisfiescc(point, cc) ):
                # print("Am I stuck here?") # Sampling efficiency for affine spaces is too low.
                for i in range(n):
                    point[i] = random.randint(minpoints[i], maxpoints[i] + 1)     
            points.append(point) 
        return points      
    
    n  = len(cc[0]) - 2
    if (isAffine(cc)):
        (A, b, nonA, nonb) = getAffine(cc) #Adds the Dstate requirement to the nonAffine predicates too.    
        (basevector, colvectors) = linearDiophantineSolution(np.array(A, ndmin = 2), np.array(b))
        if (np.any(basevector)): #No solution
            return []
        S = np.transpose(np.array(colvectors, ndmin = 2))
        if (colvectors == []): 
            return [basevector]
        A2 = np.matmul(nonA, S)
        b2 = nonb - np.matmul(nonA, basevector)
        
        lambda_cc = []
        (A2_rows, _) = A2.shape
        for i in range(A2_rows):
            lambda_cc.append( [int(x) for x in A2[i]  ]  + [-1, b2[i]])   # Original RHS is [int(A2[i]), -1, b2[i]])   
        endpoints = v_representation(np.array(lambda_cc, ndmin = 2, dtype = int))
        
        
        lambdapoints = randomlysamplepoints( endpoints, np.array(lambda_cc, ndmin = 2, dtype = int))
                    
        return [ (basevector + np.matmul(S, v)).tolist() for v in lambdapoints]
    else:
        endpoints = v_representation(dnfconjunction([cc], Dstate(n), 0)[0])
        if endpoints == []:
            return []
        return randomlysamplepoints(endpoints, dnfconjunction([cc], Dstate(n), 0)[0])







# Don't need this any more!
# def ExtremeLatticePoints( v_repr, cc, n):
#     def nonintegralcoordinates( pt):
#         rv = []
#         for i in range(n):
#             if not isinstance(pt[i], int):
#                 rv.append(i)
#         return rv

#     def binarylist(n, size):
#         binaryNumber = [int(x) for x in bin(n)[2:]]
#         paddedSize = size - len(binaryNumber)
#         return [0] * paddedSize + binaryNumber

#     rv = []
#     for pt in v_repr:
#         nonIntegerCoordinates = nonintegralcoordinates(pt)
#         if (len(nonIntegerCoordinates) == 0):
#             rv.append(pt)
#         else:
#             d = len(nonIntegerCoordinates)
#             for i in range(1, 2**d + 1):
#                 pt_new = pt.copy()
#                 binaryList = binarylist(i, d)
#                 for i in range(d):
#                     if (binaryList[i] == 0):
#                         pt_new[nonIntegerCoordinates[i]] = floor(pt_new[nonIntegerCoordinates[i]])
#                     else:
#                         pt_new[nonIntegerCoordinates[i]] = ceil(pt_new[nonIntegerCoordinates[i]])    
#                 if (pointsatisfiescc(pt_new, cc)):
#                     rv.append(pt_new)
#     return rv            

# def get_cc_pts (cc, m):
#     n = len(cc[0]) - 2
#     v_repr = v_representation(cc)
#     vol = getvolume(v_repr, n)
#     # r = getmaxradius (v_repr)
#     if (vol > conf.SmallVolume):
#         return randomlysamplepointsCC(cc, m)
#     else:
#         return ExtremeLatticePoints(v_repr, cc, n)

    #Need to store initial computation, and then finally do for multiple points in 1 go.


def pointsatisfiesdnf(pt, dnf):
    def pointsatisfiescc (pt, cc):
        for p in cc:
            sat = sum(p[:-2]* pt) - p[-1]
            if (sat > 0):
                return False
        return True  
    for cc in dnf:
        if (pointsatisfiescc(pt, cc)):
            return True
    return False      


def TIterates (T):
    def rtfIterates (rtf):
        def ptfIterates (ptf):
            T_curr = ptf
            T_list = [np.copy(T_curr)]

            maxiterate = max(conf.maxiterateICE, conf.maxiterateImplPair)
            
            for _ in range(2, maxiterate + 1):
                T_curr = T_curr @ ptf #Matrix multiplication
                T_list.append(np.copy(T_curr))   
            
            return T_list

        rv = []
        for ptf in rtf.tlist:
            rv.append(ptfIterates(ptf))
        
        return rv    

    rv = []
    for rtf in T:
        rv.append(rtfIterates(rtf))
    
    return rv
        






def iteratedtransitions (x , ptf, B, maxiterate, T_list):
    # T_curr = ptf
    # T_list = []

    # for _ in range(2, maxiterate + 1):
    #     T_curr = T_curr @ ptf #Matrix multiplication
    #     T_list.append(np.copy(T_curr))
    
    i = 0 
    while (i < len(T_list)):
        y = transition(x , T_list[i])
        if (not pointsatisfiesdnf(y, B)):
            break
        i = i + 1
    i = i - 1
    
    # print(i) ##Debug
    if (i <= 0):
        return []
    return transition(x, T_list[i])


def filter_ICEtails(n, tl_list):
    rv = []
    for tl in tl_list:
        if ( LIDNFptdistance(Dstate(n), tl) == 0):
            rv.append(tl)
    return rv


#CC is 2d numpy array; unbounded polytopes allowed
def randomlysampleCC_ICEpairs (cc, m, transitions, loopGuard, rtfIterates):
    def pointsatisfiescc (pt, cc):
        for p in cc:
            sat = sum(p[:-2]* pt) - p[-1]
            if (sat > 0):
                return False
        return True

    n  = len(cc[0]) - 2
    if (isAffine(cc)):     
        (A, b, nonA, nonb) = getAffine(cc) #Adds the Dstate requirement to the nonAffine predicates too.
        (basevector, colvectors) = linearDiophantineSolution(np.array(A, ndmin = 2), np.array(b))
        if (basevector == []): #No solution
            return []
        if (colvectors == []):
            tls = filter_ICEtails(n, [ transition(basevector, ptf) for ptf in transitions])  
            return [ (basevector, tl) for tl in tls] 
        S = np.transpose(np.array(colvectors, ndmin = 2))
        A2 = np.matmul(nonA, S)
        b2 = nonb - np.matmul(nonA, basevector)        
        
        lambda_cc = []
        (A2_rows, A2_columns) = A2.shape
        for i in range(A2_rows):
            lambda_cc.append( [int(x) for x in A2[i]  ]  + [-1, b2[i]])   # Original RHS is [int(A2[i]), -1, b2[i]]) 
        
        
        
        endpoints = v_representation(np.array(lambda_cc, ndmin = 2, dtype = int))
        minpoints = [ conf.dspace_intmax] * A2_columns
        maxpoints = [ conf.dspace_intmin] * A2_columns
        for pt in endpoints:
            for i in range(A2_columns):
                A = floor(pt[i])
                B = ceil(pt[i])
                if( B < minpoints[i]):
                    minpoints[i] = B
                if( A > maxpoints[i]):
                    maxpoints[i] = A
        
        rv = []
        #ICE star outputted
        for i in range(m):
            samplepoint = [conf.dspace_intmin - 2]*A2_columns
            tls = []
            while ( tls == [] ):
                # print(basevector, S , samplepoint) #Debugging
                point = (basevector + np.matmul(S, samplepoint)).tolist()
                while (  not pointsatisfiescc(point, np.array(lambda_cc, ndmin = 2, dtype = int)) ):
                    for i in range(A2_columns):
                        samplepoint[i] = random.randint(minpoints[i], maxpoints[i] + 1)     
                    point = (basevector + np.matmul(S, samplepoint)).tolist()
                tls = filter_ICEtails(n, [ transition(point, ptf) for ptf in transitions])   
                longtls = []
                for (i,ptf) in enumerate(transitions):
                    newtl = iteratedtransitions(point, ptf, loopGuard, conf.maxiterateImplPair, rtfIterates[i])
                    if (newtl == []):
                        continue
                    longtls.append(newtl)
                tls = tls + filter_ICEtails(n, longtls)
                                          
            rv = rv + [ (point, tl) for tl in tls]            
        return rv
    else:
        n  = len(cc[0]) - 2
        endpoints = v_representation(dnfconjunction([cc], Dstate(n), 0)[0])
        
        
        
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
        for i in range(m):
            point = [0]*n
            tls = []
            while ( tls == [] ):
                for i in range(n):
                    point[i] = random.randint(minpoints[i], maxpoints[i] + 1)  

                if (not pointsatisfiescc(point, cc)):
                    continue
                tls = filter_ICEtails(n, [ transition(point, ptf) for ptf in transitions])
                longtls = []
                for (i,ptf) in enumerate(transitions):
                    newtl = iteratedtransitions(point, ptf, loopGuard, conf.maxiterateImplPair, rtfIterates[i])
                    if (newtl == []):
                        continue
                    longtls.append(newtl)
                tls = tls + filter_ICEtails(n, longtls)                           
            rv = rv + [ (point, tl) for tl in tls]
        return rv



# def get_cc_ICEheads (cc, m, transitions):
#     n = len(cc[0]) - 2
#     v_repr = v_representation(cc)
#     vol = getvolume(v_repr, n)
#     # r = getmaxradius (v_repr)
#     if (vol > conf.SmallVolume):
#         return randomlysampleCC_ICEpairs(cc, m, transitions)
#     else:
#         rv = []
#         latticeVrepr = ExtremeLatticePoints(v_repr, cc, n)
#         for hd in latticeVrepr:
#             tls = filter_ICEtails(n, [ transition(hd, ptf) for ptf in transitions])
#             if (tls == []):
#                 continue
#             for tl in tls:
#                rv.append( (hd, tl)) 
#         return rv


def removeduplicates (l):
    L = copy.deepcopy(l)
    L.sort()
    return list(L for L,_ in itertools.groupby(L))

def removeduplicatesICEpair(input_list):
    seen_pairs = set()
    result = []
    
    for pair in input_list:
        pair_tuple = (tuple(pair[0]), tuple(pair[1]))
        if pair_tuple not in seen_pairs:
            result.append(pair)
            seen_pairs.add(pair_tuple)
    
    return result


def get_plus0 (P, m):
    n = len(P[0][0]) - 2  

    plus0 = []
    
    P_LIA = genLII_to_LII(P)
    
    for cc in P_LIA:
        if (isEmpty(cc)):
            continue        
        
        plus0 = plus0 + randomlysamplepointsCC(cc, m)


    return removeduplicates(plus0)  

def get_minus0 (Q, m):
    n = len(Q[0][0]) - 2
    negQ = dnfnegation(Q) #This always returns LIA form

    minus0 = []
    for cc in negQ:
        
        if (isEmpty(cc)):
            continue

        
        minus0 = minus0 + randomlysamplepointsCC( dnfconjunction([cc], Dstate(n) , 0)[0], m) #Need a separate conjunction here as sampling from ~Q


    return removeduplicates(minus0)


def get_ICE0( T, P, Q, m, Titerates):
    n = len(T[0].b[0][0]) - 2
    rv = []
    def partialICE_enet (B , P, Q, transitionlist, m, n, rtfIterates):
        rv = []
        #The space is B 
        for cc in B:
            if (isEmpty(cc)):
                continue
            rv = rv + randomlysampleCC_ICEpairs(cc, m, transitionlist, B, rtfIterates)
        return rv

    for (i,rtf) in enumerate(T):
        rv = rv + partialICE_enet (rtf.b , P, Q, rtf.tlist, m, n, Titerates[i]) #All rft.b are in LIA form
    

    
    return removeduplicatesICEpair(rv)

       

def get_longICEpairs( ICE, T, n, Titerates):
    rv = []
    for (hd, _) in ICE:
        tls = []
        for (i,rtf) in enumerate(T):
            if ( pointsatisfiesdnf(hd, rtf.b) ):
                longtls = []
                for (j,ptf) in enumerate(rtf.tlist):
                    newtl = iteratedtransitions(hd, ptf, rtf.b, conf.maxiterateICE, Titerates[i][j])
                    if (newtl == []):
                        continue
                    longtls.append(newtl)
                tls = tls + filter_ICEtails(n, longtls)
                break
        for newtl in tls:
            rv.append((hd,newtl))
    
    return rv

# print( randomlysamplepointsCC( np.array( [[1, 0, 1, 0] , [1, 0, -1, 2] , [0, 1, 1, 0] , [0, 1, -1, 2]]   ) , 8) )