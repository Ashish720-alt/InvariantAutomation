""" Guessing a new invariant. """

import numpy as np
from configure import Configure as conf
from math import inf, sqrt, sin, log , ceil, e, floor
from cost_funcs import cost
from dnfs_and_transitions import  list3D_to_listof2Darrays, dnfconjunction
from repr import Repr

def COR_FP_Origin(oldcoeff, oldconst, newcoeff, k1):
    symconsts = []
    dotproduct = 1.0 * np.dot(np.array(newcoeff), np.array(oldcoeff))
    oldnorm = 1.0 * np.dot(np.array(oldcoeff), np.array(oldcoeff))
    newnorm = 1.0 * np.dot(np.array(newcoeff), np.array(newcoeff)) 
    asymconst = floor( oldconst * dotproduct/ oldnorm  )
    symconstmin = max(ceil(oldconst * newnorm / dotproduct), ceil(-1* k1) )
    symconstmax = min(ceil( (oldconst + 1) * newnorm / dotproduct), ceil(k1)) 
    symconsts = list(range(symconstmin, symconstmax + 1))
    if asymconst not in symconsts and asymconst <= k1 and asymconst >= -k1:
        symconsts.append(asymconst)
    return symconsts

def LargeTranslationConstant(oldcoeff, oldconst, k1):
    slope = sqrt(np.dot(np.array(oldcoeff), np.array(oldcoeff)))
    translation_range = floor(conf.translation_range * slope)
    max_pos_dev = min(floor(k1 - oldconst), floor(translation_range))
    max_neg_dev = max(ceil(-k1 - oldconst), ceil(-translation_range) ) 
    translation_indices = list(range(oldconst + max_neg_dev, oldconst + max_pos_dev +1))
    translation_indices.remove(oldconst)
    return translation_indices

def rotation_neighbors(i, j, coeff, const, repr: Repr, k1, n, tID):
    if (conf.BASIC_ROTATION == conf.OFF):
        if (n <= 3):
            rotneighbors = repr.get_coeffneighbors(coeff)
        else:
            negative_indices = [idx for idx, val in enumerate(coeff) if val < 0]
            rotneighbors = [ [-coeff[i] if i in negative_indices else coeff[i] for i in range(n)]  
                                    for coeff in repr.get_coeffneighbors([abs(val) for val in coeff])]
        rv = []
        for r in rotneighbors:
            if (conf.COR_SIMPLIFIED == conf.OFF):
                constlist = COR_FP_Origin(coeff, const, r, k1)
            else:
                constlist = [const] #Using the same constants as before, as cost change is barely much.
            for co in constlist:
                rv.append( ( i, j, r + [-1,co]) )
        return rv
    else:
        rv = []
        unallowedcoeff = [0]*len(coeff)
        for k in range(len(coeff)):
            newcoeff1 = coeff.copy()
            newcoeff2 = coeff.copy()
            newcoeff1[k] = newcoeff1[k] + 1
            newcoeff2[k] = newcoeff2[k] - 1 
            if (newcoeff1[k] <= conf.BASIC_ROTATION_k0List[tID] and newcoeff1 != unallowedcoeff):
                rv.append((i,j, newcoeff1 + [-1, const]))
            if (newcoeff2[k] >= -conf.BASIC_ROTATION_k0List[tID] and newcoeff2 != unallowedcoeff):
                rv.append((i,j, newcoeff2 + [-1, const]))
        return rv

def translation_neighbors(i, j, coeff, const, k1):
    rv = []
    if (conf.TRANSLATION_SMALL == conf.OFF ):
        transconslist = LargeTranslationConstant(coeff, const, k1)
    else:
        # if (const == 10):
        #    transconslist = [const - 1]
        # elif (const == -10):
        #    transconslist = [const + 1]
        #else:    
        transconslist = [const + 1, const - 1] 
    for co in transconslist:
        rv.append( ( i, j, coeff + [-1,co]) )  
    return rv

def theoreticalSAconstantlist( TS, k0, n, c, d, k1_list):    
    def SAconstant(TS_size, k0, k1, n , c , d):
        L_upper = 4 * TS_size * max(conf.translation_range, 4 * k1 * sqrt(n) * sin( conf.rotation_rad / 2 ) )
        r_upper = c * d * 2 * k0 * k1 / ( (k0 - 1) * conf.translation_range ) 
        return L_upper * r_upper

    return [SAconstant( TS, k0, k1 , n, c, d ) for k1 in k1_list]  


def FasterBiased_RWcostlist(tID, I, samplepoints, repr): 
    n = repr.get_n()
    c1 = float('inf')
    c2 = 0
    rv = []
    
    counter = 0
    while (len(rv) < conf.T0_COSTLISTLENGTH):    
        if (counter >= 3*conf.T0_COSTLISTLENGTH):
            rv.append((0,1))
            return rv
        LII = dnfconjunction(list3D_to_listof2Darrays(I), repr.get_affineSubspace() , 0)
        (c2, _) = cost(LII, samplepoints)  
        if (c2 > c1): #only positive transitions
            rv.append((c1,c2))
        i = np.random.randint(0, repr.get_d())
        j = np.random.randint(0, repr.get_cList()[i])
        if (np.random.rand() < conf.T0_rotationMaxProb): #Choose rotation or translation
            neighbors = rotation_neighbors(i, j, I[i][j][:-2], I[i][j][-1], repr, repr.get_k1(), n, tID)
        else:
            neighbors = translation_neighbors(i, j, I[i][j][:-2], I[i][j][-1], repr.get_k1())
        I[i][j] = neighbors[np.random.choice(range(len(neighbors)))][2]
        c1 = c2
        counter = counter + 1
        
    return rv 

def experimentalSAconstantlist(Ilist, samplepoints, repr):
    def ExponentialWithError(x, y):
        try:
            result = e**(-x / y)
        except OverflowError:
            result = 0
        except FloatingPointError:
            result = 0
        return result
    rv = []
    for (tID,I0) in enumerate(Ilist):
        S = FasterBiased_RWcostlist(tID, I0, samplepoints, repr)    
        T = -sum( [ i[1] - i[0] for i in S ] ) / (log(conf.T0_X0) * len(S))
        N = [ ExponentialWithError(i[1], T) for i in S ] 
        D = [ ExponentialWithError(i[0], T) for i in S ]
        default_positive_transition = (0, 30*T) #The minimum value which python represents is 2.22 * 1e-16, which holds for a cost change of 36.04*T
        if sum(D) == 0:  #To avoid division by zero for very small cost changes, add the default positive transition 
            N.append(ExponentialWithError(default_positive_transition[1], T))
            D.append(ExponentialWithError(default_positive_transition[0], T))
            S.append(default_positive_transition)
        X_T = sum(N) / sum(D) 
        while (  abs(X_T - conf.T0_X0) > conf.T0_e ):
            T = T * (log(X_T)/ log(conf.T0_X0))
            N = [ ExponentialWithError(i[1], T) for i in S ]
            D = [ ExponentialWithError(i[0], T) for i in S ]
            default_positive_transition = (0, 35*T) 
            if sum(D) == 0:  #T decreases in this loop, hence at some point D may add up to zero even if it wasn't previously doing so.
                N.append(ExponentialWithError(default_positive_transition[1], T))
                D.append(ExponentialWithError(default_positive_transition[0], T))
                S.append(default_positive_transition)
            X_T = sum(N) / sum(D)    
        rv.append(T)             
    return rv

def randomlysampleelementfromList(l):
    return l[np.random.choice( len(l))]

def uniformlysample_I( tID, rotation_vertices, k1, cList, d, n, Dp):
    def uniformlysample_cc(tID, rotation_vertices, k1, n, c, Dp):
        def uniformlysample_p(tID, rotation_vertices, k1, n, Dp):
            if (conf.BASIC_ROTATION == conf.OFF):
                coeff = randomlysampleelementfromList(rotation_vertices)
            else:
                unallowedcoeff = [0]*n
                coeff = unallowedcoeff
                while (coeff == unallowedcoeff):
                    coeff = []
                    for _ in range(n):
                        # coeff.append(np.random.choice(range(-conf.BASIC_ROTATION_k0, conf.BASIC_ROTATION_k0+1)))
                        coeff.append(np.random.choice(range(-conf.BASIC_ROTATION_k0List[tID], conf.BASIC_ROTATION_k0List[tID]+1)))
            if (n > 1):
                const = 0 # This gives  better results
            else:
                const = randomlysampleelementfromList(Dp)
            return list(coeff) + [-1,const]
        return  [ uniformlysample_p(tID, rotation_vertices, k1, n, Dp) for _ in range(c)  ]
    return [ uniformlysample_cc(tID, rotation_vertices, k1, n, ci, Dp) for ci in cList  ]


def initialInvariant( tID, samplepoints, rotation_vertices, k1, cList, d, n, affinespace, Dp):
    I = []
    costI = inf
    samplesize = conf.I0_samples if (n > 1) else conf.I0_samples_n1
    for _ in range(samplesize):
        Inew = uniformlysample_I(tID, rotation_vertices, k1, cList, d, n, Dp)
        LII = dnfconjunction( list3D_to_listof2Darrays(Inew), affinespace , 0)
        (costInew, _ ) = cost(LII, samplepoints)
        if (costInew < costI):
            I = Inew
            costI = costInew
    
    return (I, costI)



def SearchSpaceNeighbors(I, repr: Repr, d, cList, k1, n, tID):
    neighbors = []
    for i in range(d):
        for j in range(cList[i]):
            oldcoeff = I[i][j][:-2]
            oldconst = I[i][j][-1]
            if (conf.GUESS_SCHEME != conf.ONLY_TRANSLATION):
                neighbors = neighbors + rotation_neighbors(i, j, oldcoeff, oldconst, repr, k1, n, tID)
            if (conf.GUESS_SCHEME != conf.ONLY_ROTATION):
                neighbors = neighbors + translation_neighbors(i, j, oldcoeff, oldconst, k1)
    return neighbors  


# def guessInvariants( samplepoints, rotation_vertices, k1, c, d, n, affinespace, Dp, ct):
#     population = []
#     for _ in range(ct):
#         I = uniformlysample_I( rotation_vertices, k1, c, d, n, Dp)
#         LII = dnfconjunction( list3D_to_listof2Darrays(I), affinespace , 0)
#         (costI, _ ) = cost(LII, samplepoints)
#         population.append((I, costI))
    
#     return population



