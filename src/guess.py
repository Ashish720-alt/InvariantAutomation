""" Guessing a new invriant.
"""
import numpy as np
from configure import Configure as conf
from math import inf, sqrt, floor, sin, ceil, log , e
from cost_funcs import cost
from dnfs_and_transitions import  list3D_to_listof2Darrays, dnfconjunction

def k1list(k0, n):
    # return [10]
    max_radius = conf.dspace_radius * k0 * n
    
    n = floor(conf.num_processes/2)
    radius_list = [1000] * n + [max_radius] * (conf.num_processes - n)
        
    return radius_list   


def SAconstantlist( TS, k0, n, c, d, k1_list):    
    def SAconstant(TS_size, k0, k1, n , c , d):
        L_upper = conf.beta * TS_size * max(conf.translation_range, 4 * k1 * sqrt(n) * sin( conf.rotation_rad / 2 ) )
        r_upper = c * d * 2 * k0 * k1 / ( (k0 - 1) * conf.translation_range ) 
        return L_upper * r_upper

    return [SAconstant( TS, k0, k1 , n, c, d ) for k1 in k1_list]  


def experimentalSAconstantlist():
    S = []
    n =  conf.S_changecostmax
    for Emin in range(conf.S_Maxcost):
        deltamax = floor(Emin * n)
        for delta in range(1, deltamax + 1):
            S.append([Emin,Emin + delta])
    
    T = sum( [ i[1] - i[0] for i in S ] ) / (log(conf.T0_X0) * len(S))
    X_T = sum([ e**(- i[1]/ T) for i in S ]) / sum([ e**(- i[0]/ T) for i in S ])
    while (  abs(X_T - conf.T0_X0) > conf.T0_e ):
        T = T * (log(X_T)/ log(conf.T0_X0))
        X_T = sum([ e**(- i[1]/ T) for i in S ]) / sum([ e**(- i[0]/ T) for i in S ])
    
    return [T for _ in range(conf.num_processes)]

def randomlysamplelistoflists(l):
    return l[np.random.choice( len(l))]


def uniformlysample_I( rotation_vertices, k1, c, d, n):
    def uniformlysample_cc(rotation_vertices, k1, n, c):
        def uniformlysample_p(rotation_vertices, k1, n):
            coeff = randomlysamplelistoflists(rotation_vertices)
            # const = np.random.choice( list(range(-k1-1, k1 + 1 )) )
            const = 0 # This gives  better results
            return list(coeff) + [-1,const]
        return  [ uniformlysample_p(rotation_vertices, k1, n) for i in range(c)  ]
    return [ uniformlysample_cc(rotation_vertices, k1, n, c) for i in range(d)  ]


def initialInvariant( samplepoints, rotation_vertices, k1, c, d, n, affinespace):
    I = []
    costI = inf
    for _ in range(conf.I0_samples):
        Inew = uniformlysample_I( rotation_vertices, k1, c, d, n)
        LII = dnfconjunction( list3D_to_listof2Darrays(Inew), affinespace , 0)
        (costInew, _ ) = cost(LII, samplepoints)
        if (costInew < costI):
            I = Inew
            costI = costInew
    
    return (I, costI)

def rotationdegree(rotationneighbors):
    return len(rotationneighbors)

# C = op_norm_conj(C)
# A = np.concatenate(
#     [C[:, :self.num_var], C[:, self.num_var+1:]], axis=1)
# return float(minimize(
#     lambda x, p: np.linalg.norm(x - p),
#     np.zeros(self.num_var),
#     args=(p,),
#     constraints=[LinearConstraint(A[:, :-1], -np.inf, -A[:, -1])],
# ).fun)

def listmultiplyconstant(c, l):
    return [ c * x for x in l ]

def listadd(l1, l2):
    return [sum(p) for p in zip(l1, l2)]

# # Here, posed as an ILP!
# def centre_of_rotation_new(pred, newcoefficient, spin, k1):
#     # Need to convert the type of elements of this array to float?
#     coeff = pred[:-2]
#     const = pred[-1]
#     n = len(coeff)
#     sign = 1 if (spin >= 0) else -1
#     v = listadd(listmultiplyconstant( 1.0/ sqrt(np.dot(newcoefficient, newcoefficient)) , newcoefficient) ,listmultiplyconstant(-1.0/ sqrt(np.dot(coeff, coeff)), coeff))
    
#     # print(coeff, const, n, sign, v) #Debugging
    
#     return minimize(
#         lambda x, v, spin: spin * np.dot(np.array(v), np.array(x)),
#         np.zeros(n), #This is the initial guess!
#         args=(v,spin),
#         bounds = Bounds(lb = np.full(n, conf.dspace_intmin), ub = np.full(n, conf.dspace_intmax) ),
#         # Without newconstant constraints
#         # constraints=[LinearConstraint(np.array( [ coeff, listmultiplyconstant(-1, coeff) ]  ), np.array( [-np.inf, -np.inf] ), np.array( [const, -const] ) )],
#         # With new constant constraints
#         constraints=[LinearConstraint(np.array( [ coeff, listmultiplyconstant(-1, coeff) , newcoefficient]  ), np.array( [-np.inf, -np.inf, -k1 - 1] ), np.array( [const, -const, k1] ) )], #Convert this to shape (1,n) instead of (n)?
#     ).x

# print(centre_of_rotation_new( [-1,2,-1,200] , [-1,1] , 1 ))

def origin_fp(pred):
    coeff = pred[:-2]
    const = 1.0 * pred[-1] #This should be plus as our invariant is ax + by - c <= 0 and fp is (h-x1)/a = (k - y1)/b = -(ax1 + by1 + c)/ (a^2 + b^2)
    K = np.dot(np.array(coeff) , np.array(coeff))
    return [ (x * const * 1.0)/K for x in coeff ]


# def centre_of_rotation_projectedWalk(pred, filteredpoints):
#     coeff = pred[:-2]
#     o_fp = origin_fp(pred)
#     const = round(np.dot(np.array(coeff), np.array(o_fp)), 0) 
#     return const


# def centre_of_rotation_walk(pred, filteredpoints):
#     n = len(pred) - 2
#     coeff = pred[:-2]
#     A = np.array([ np.array(coeff) for i in range(n)])
#     ns_columnarray = null_space(A)
#     ns_array = np.transpose(ns_columnarray)
#     ns_list = [list(x) for x in ns_array] 

#     def coordinate_bounds(basis):
#         def coeffbounds(basisvector):
#             (maxvalue, minvalue) = (max(basisvector), min(basisvector))
#             (maxposvalue, minnegvalue) = (max(0.01, maxvalue), min(-0.01, minvalue) )
#             U = min( conf.dspace_intmax/(maxposvalue), conf.dspace_intmin/(minnegvalue)  )
#             L = max( conf.dspace_intmax/(minnegvalue), conf.dspace_intmin/(maxposvalue)  )
#             return (L,U)
#         return [coeffbounds(v) for v in basis]

#     bounds = coordinate_bounds(ns_list)
#     K = pred[-1] / np.dot(np.array(coeff), np.array(coeff))

#     def coordinate_to_point(coordinates, basis_list, K): #ns_list is basis_list
#         rv = np.zeros(n, dtype = float)
#         for i in range(len(basis_list)):
#             rv = np.add(rv, coordinates[i]*np.array(basis_list[i]) )
#         rv = np.add(rv, K*np.array(coeff))
#         return list(rv)

#     def centreofrotation_cost(newI, filteredpoints):
#         (pos_cost, _, _) = costplus(newI, filteredpoints[0])
#         (neg_cost, _, _) = costminus(newI, filteredpoints[1])
#         (ICE_cost, _, _) = costICE(newI, filteredpoints[2])
#         return pos_cost + neg_cost + ICE_cost


#     # x0 = origin_fp(pred)
#     # coordinate_curr =  list(np.transpose(np.matmul( inv(ns_columnarray) , np.transpose(np.array(x0)) )))
#     k = len(ns_list)
#     coordinate_curr = [0] * k
#     point_curr = coordinate_to_point(coordinate_curr, ns_list, K)
#     curr_const = round(np.dot(np.array(coeff), np.array(point_curr)), 0) 
#     curr_cost = centreofrotation_cost([np.array(coeff + [-1, curr_const], ndmin = 2)], filteredpoints)
#     i = 1
#     while (i < conf.centre_walklength):
#         j = random.choice(list(range(k)))
#         change = random.choice([-1,1])
#         coordinate_temp = coordinate_curr.copy()
#         coordinate_temp[j] = coordinate_temp[j] + change
#         print(coordinate_curr, coordinate_temp) #Debug
#         point_temp = coordinate_to_point(coordinate_temp, ns_list, K)
#         temp_const = round(np.dot(np.array(coeff), np.array(point_temp)), 0) 
#         temp_cost = centreofrotation_cost([np.array(coeff + [-1, temp_const], ndmin = 2)], filteredpoints)
#         if (temp_cost < curr_cost):
#             print("Walk has moved!") #Debug
#             curr_cost = temp_cost
#             coordinate_curr = coordinate_temp
#             point_curr = point_temp
#         i = i + 1
#     return point_curr

# # Uniformly samples a point (not necessarily lattice point) on the hyperplane upto some approximation error (usually 1e-9)
# def centre_of_rotation(pred):
#     n = len(pred) - 2
#     coeff = pred[:-2]
#     A = np.array([ np.array(coeff) for i in range(n)])
#     ns_array = np.transpose(null_space(A))
#     ns_list = [list(x) for x in ns_array] 

#     def coordinate_bounds(basis):
#         def coeffbounds(basisvector):
#             (maxvalue, minvalue) = (max(basisvector), min(basisvector))
#             (maxposvalue, minnegvalue) = (max(0.01, maxvalue), min(-0.01, minvalue) )
#             U = min( conf.dspace_intmax/(maxposvalue), conf.dspace_intmin/(minnegvalue)  )
#             L = max( conf.dspace_intmax/(minnegvalue), conf.dspace_intmin/(maxposvalue)  )
#             return (L,U)
#         return [coeffbounds(v) for v in basis]

#     bounds = coordinate_bounds(ns_list)

#     def samplepoint(basis, bounds, coeff, b):
#         def isvalidvector(pt):
#             return all([ ((val >= conf.dspace_intmin) and (val <= conf.dspace_intmax)) for val in pt ])
#         n = len(basis[0])
#         rv = [conf.dspace_intmax+1]*n
#         K = b / np.dot(np.array(coeff), np.array(coeff))
#         while( not isvalidvector(list(rv))):
#             coordinates = [ np.random.uniform(I[0], I[1]) for I in bounds] #Uniform Sampling
#             # coordinates = [ np.random.normal(loc=0.0, scale=1.0) for I in bounds] #Gaussian Sampling
#             rv = np.zeros(n, dtype = float)
#             for i in range(len(basis)):
#                 rv = np.add(rv, coordinates[i]*np.array(basis[i]) )
#             rv = np.add(rv, K*np.array(coeff))
#         return list(rv)
    
#     return [ samplepoint(ns_list, bounds, coeff, pred[-1]) for x in range(conf.centres_sampled)]

# def centre_of_rotation_old(oldpredicate, filteredpoints, newcoefficient):
#     centreofrotation_list = centre_of_rotation(newcoefficient + [-1,0])

#     def centreofrotation_cost(newI, filteredpoints):
#         (pos_cost, _, _) = costplus(newI, filteredpoints[0])
#         (neg_cost, _, _) = costminus(newI, filteredpoints[1])
#         (ICE_cost, _, _) = costICE(newI, filteredpoints[2])
#         return pos_cost + neg_cost + ICE_cost

#     centreofrotation = centreofrotation_list[0]
#     const = round(np.dot(np.array(newcoefficient), np.array(centreofrotation)), 0) 
#     newpred = newcoefficient + [-1, const]
#     newI = [np.array( newpred, ndmin = 2)]
#     cost = centreofrotation_cost(newI, filteredpoints)

#     i = 1
#     while (i < conf.centres_sampled):
#         curr_centreofrotation = centreofrotation_list[i]
#         curr_const = round(np.dot(np.array(newcoefficient), np.array(curr_centreofrotation)), 0) 
#         currpred = newcoefficient + [-1, curr_const]
#         currI = [np.array( currpred, ndmin = 2)]
#         curr_cost = centreofrotation_cost(currI, filteredpoints)
#         if (curr_cost < cost):
#             cost = curr_cost
#             centreofrotation = curr_centreofrotation
#         i = i+1
#     return centreofrotation

# #samplepoints is a triple, costlist is a single list
# def getrotationcentre_points(samplepoints, costlists, oldpred):
#     pos_costlist = costlists[0: len(samplepoints[0])]
#     neg_costlist = costlists[len(samplepoints[0]) : len(samplepoints[0]) + len(samplepoints[1])]
#     ICE_costlist = costlists[len(samplepoints[0]) + len(samplepoints[1]): ]

#     neg_oldpred = [-1*x for x in oldpred]  
#     neg_oldpred[-2] = -1
#     neg_oldpred[-1] = neg_oldpred[-1] - 1

#     positivepts = samplepoints[0]
#     negativepts = samplepoints[1]
#     ICEpts = samplepoints[2]

#     filtered_pos = []
#     filtered_neg = []
#     filtered_ICE = []

#     def pt_linedistance(pred, pt):
#         magnitude = sqrt(sum(i*i for i in pred[:-2]))
#         s = 0
#         for i in range(len(pt)):
#             s = pred[i]*pt[i]
#         s = s - pred[-1]    
#         return  s/magnitude

#     for i,pos in enumerate(positivepts):
#         dist = pt_linedistance(oldpred, pos)
#         if ( dist > 0 and dist <= conf.d and pos_costlist[i] > 0):
#             filtered_pos.append(pos)

#     for i,neg in enumerate(negativepts):
#         dist = pt_linedistance(neg_oldpred, neg)
#         if ( dist > 0 and dist <= conf.d and neg_costlist[i] > 0):
#             filtered_neg.append(neg)

#     for i,ICE in enumerate(ICEpts):
#         dist1 = pt_linedistance(neg_oldpred, ICE[0])
#         dist2 = pt_linedistance(oldpred, ICE[1])
#         if ( dist1 <= 0 and dist2 > 0 and min(-dist1, dist2) <= conf.d and ICE_costlist[i] > 0):
#             filtered_ICE.append(ICE)

#     return (filtered_pos, filtered_neg, filtered_ICE )


# whether rotation is possible or not i.e. if while loop is not an infinite loop is handled in isrotationchange(.. , .. , ..) function
def rotationtransition(oldpredicate, rotationneighbors, k1):
    
    # n = len(oldpredicate) - 2
    oldcoeff = oldpredicate[:-2]
    oldconst = oldpredicate[-1]    
    newcoeff = oldpredicate[:-2]
    newconst = inf
    
    while (newconst > k1 or newconst < -1 * k1):
        newcoeff = list(randomlysamplelistoflists(rotationneighbors)) 
        newconst = floor( oldconst * (1.0 * np.dot(np.array(newcoeff), np.array(oldcoeff)))/ (1.0 * np.dot(np.array(oldcoeff), np.array(oldcoeff)))  )

    return newcoeff + [-1, newconst]

def getNewRotConstant(oldcoeff, oldconst, newcoeff, k1):
    symconsts = []
    dotproduct = 1.0 * np.dot(np.array(newcoeff), np.array(oldcoeff))
    oldnorm = 1.0 * np.dot(np.array(oldcoeff), np.array(oldcoeff))
    newnorm = 1.0 * np.dot(np.array(newcoeff), np.array(newcoeff)) 
    asymconst = floor( oldconst * dotproduct/ oldnorm  )
    symconstmin = max(ceil(oldconst * newnorm / dotproduct), -1* k1)
    symconstmax = min(ceil( (oldconst + 1) * newnorm / dotproduct), k1)
    symconsts = list(range(symconstmin, symconstmax + 1))
    if asymconst not in symconsts and asymconst <= k1 and asymconst >= -k1:
        symconsts.append(asymconst)
    return symconsts
    
    


def translationtransition(predicate, k1):
    slope = sqrt(np.dot(np.array(predicate[:-2]), np.array(predicate[:-2])))
    translation_range = floor(conf.translation_range * slope)
    translation_indices = list(range(-translation_range, translation_range +1))
    translation_indices.remove(0)
    s = inf
    rv = predicate.copy()
    while (rv[-1] + s > k1 or rv[-1] + s < -1 * k1):
        s = np.random.choice(translation_indices)
    rv[-1] = rv[-1] + s
    return rv

def getNewTranslationConstant(oldcoeff, oldconst, k1):
    slope = sqrt(np.dot(np.array(oldcoeff), np.array(oldcoeff)))
    translation_range = floor(conf.translation_range * slope)
    max_pos_dev = min(k1 - oldconst, translation_range)
    max_neg_dev = max(-k1 - oldconst, -translation_range )
    translation_indices = list(range(oldconst + max_neg_dev, oldconst + max_pos_dev +1))
    translation_indices.remove(oldconst)
    return translation_indices


def get_index(d, c):
    return (np.random.choice( list(range(d)) ), np.random.choice( list(range(c)) ))

def isrotationchange(oldpredicate, rotationneighbors, k1):
    def isrotationpossible(oldpredicate, rotationneighbors, k1):
        oldcoeff = oldpredicate[:-2]
        oldconst = oldpredicate[-1]   
        i = 0
        for newcoeff in rotationneighbors:
            newconst = floor( oldconst * (1.0 * np.dot(np.array(newcoeff), np.array(oldcoeff)))/ (1.0 * np.dot(np.array(oldcoeff), np.array(oldcoeff)))  )
            if (not (newconst > k1 or newconst < -1 * k1)):
                return True
            i = i + 1
        if (i == len(rotationneighbors)):
            return False     
    
    return isrotationpossible(oldpredicate, rotationneighbors, k1) and (np.random.rand() <= conf.p_rot )



    



# h = [1,1,-1,3]
# a = centre_of_rotation(h)
# print(a, np.dot(np.array(h[:-2]), np.array(a)) - h[-1])