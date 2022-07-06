""" Guessing a new invriant.
"""
import numpy as np
from configure import Configure as conf
from cost_funcs import cost

# Returns a list of list of (n+1)-element-lists.
def deg_list(I, Dp):
    n = len(I[0][0]) - 2
    def degcc_list(cc, Dp, n):
        def degp_list(p, Dp, n):
            ret = []
            for i in range(n+2):
                if (i < n):
                    if (p[i] == min(Dp[0]) or p[i] == max(Dp[0]) ):
                        ret.append(1)
                    else:
                        ret.append(2)
                elif (i == n):
                    continue
                else:
                    if (p[i] == min(Dp[1]) or p[i] == max(Dp[1]) ):
                        ret.append(1)
                    else:
                        ret.append(2)
            return ret
        return [degp_list(p, Dp, n) for p in cc ]
    return [degcc_list(cc, Dp, n) for cc in I ]

def deg(deglist):
    return sum([ sum([ sum(p) for p in cc  ]) for cc in deglist])

def uniformlysampleLII(Dp, c, d, n, samplepoints):
    def uniformlysampleLIcc(Dp, n, c):
        def uniformlysampleLIp(Dp, n):
            def uniformlysamplenumber(i):
                if i < n:
                    return np.random.choice(Dp[0])
                elif i == n:
                    return -1
                else:
                    return np.random.choice(Dp[1])
            return np.fromfunction(np.vectorize(uniformlysamplenumber), shape = (n+2,), dtype=int)
        
        cc = np.empty(shape=(0,n+2), dtype = 'int')
        for i in range(c):
            cc = np.concatenate((cc, np.array([uniformlysampleLIp(Dp,n)], ndmin=2)))
        return cc
    I = [uniformlysampleLIcc(Dp, n, c) for i in range(d) ]
    (costI, mincostI, mincosttuple) = cost(I, samplepoints )
    return (I, deg_list(I, Dp), costI, mincostI, mincosttuple)


def randomwalktransition(I, deglist_I, Dp, samplepoints, prev_mincosttuple):
    # i is a number from 1 to degree
    def ithneighbor(I, i, deglist, Dp):
        k = i
        n = len(I[0][0]) - 2
        for ccindex, degcc_list in enumerate(deglist):
            for pindex, degp_list in enumerate(degcc_list):
                if ( k > sum(degp_list)):
                    k = k - sum(degp_list)
                    continue
                else:
                    index = (ccindex, pindex)
                    for vindex, deg in enumerate(degp_list):
                        if (k > deg):
                            k = k - deg
                            continue
                        else:
                            vindex_actual = vindex if (vindex < n) else (vindex + 1)                  
                            if (deg == 2):
                                I[ccindex][pindex][vindex_actual] = I[ccindex][pindex][vindex_actual] + (1 if (k == 1) else -1)
                            else: 
                                j = 0 if (vindex_actual < n+1) else 1
                                I[ccindex][pindex][vindex_actual] = I[ccindex][pindex][vindex_actual] + (1 if (I[ccindex][pindex][vindex_actual] == min(Dp[j])) else -1)
                            return (I, index)
    degree = deg(deglist_I) 
    i = np.random.choice(range(1, degree+1,1))
    (Inew, index) = ithneighbor(I, i, deglist_I, Dp)
    (costnew, mincostnew, mincosttuplenew) = cost(Inew, samplepoints, prev_mincosttuple, index)
    return (Inew, deg_list(Inew, Dp), costnew, mincostnew, mincosttuplenew)


# plus = [ [0] ]
# minus = [ [7], [10000] ]
# ICE = [ ( [5] , [6]  )  ]
# samplepoints = (plus, minus, ICE)
# (I, deglistI, costI, mincostI, mincosttupleI) = uniformlysampleLII( (range(-10, 10, 1), range(-10,10,1) ), 1, 1, 1, samplepoints )
# print(I, deglistI, costI, mincostI, mincosttupleI) 
# Dp = (range(-10, 10, 1), range(-10,10,1) )
# print(randomwalktransition(I, deglistI, Dp, samplepoints, mincosttupleI ))



