
""" Cost Functions.
This module includes cost functions.
"""

from z3 import *
import numpy as np
from repr import Repr
from dnfs_and_transitions import dnfnegation, DNF_to_z3expr, DNF_to_z3expr_p, op_norm_conj
from configure import Configure as conf
from scipy.optimize import minimize, LinearConstraint


def predicatespacedistance(p, pt):
    return 1.0

def negationpredicate(p):
    return (dnfnegation( [np.array( [p], ndmin = 2 )] ))[0][0]


def cost(I, plus, minus, ICE):
    # It is possible to optimize these computations significantly if we have the mincost_list of the previous invariant
    def mincostplus(I, plus):
        def Uplus(r):
            return 1.0
        
        def dplus(p, pluspoint):
            pluspt_array = np.array(pluspoint)
            if ( sum(p[:-2]* pluspt_array) <= p[-1] ):
                return 0
            else:
                return predicatespacedistance(p, pluspoint)

        ret = sum([ min([ sum([dplus(p, pluspoint) for p in cc  ]) for cc in I ] )  for pluspoint in plus])
        return ( ret, Uplus( ret))

    def mincostminus(I, minus):
        def Uminus(r):
            return 1.0

        def dminus(p, minuspoint):
            minuspt_array = np.array(minuspoint)
            negp = negationpredicate(p)
            if ( sum(negp[:-2]* minuspt_array) <= negp[-1] ):
                return 0
            else:
                return predicatespacedistance(negp, minuspoint)
        
        ret = sum([ min([ sum([dminus(p, minuspoint) for p in cc  ]) for cc in I ] )  for minuspoint in minus])
        return ( ret, Uminus(ret))

    def mincostICE(I, ICE):
        def UICE( r):
            return 1.0

        def dICE(p, ICEpoint):
            (hd, tl) = ICEpoint
            hd_array = np.array(hd)
            tl_array = np.array(tl)
            negp = negationpredicate(p)
            if ( ( sum(negp[:-2] * hd_array) <= negp[-1] ) or ( sum(p[:-2] * tl_array) <= p[-1] ) ):
                return 0
            else:
                return min( predicatespacedistance(negp, hd) , predicatespacedistance(p, tl)  )
        
        ret = sum([ min([ sum([dICE(p, ICEpoint) for p in cc  ]) for cc in I ] )  for ICEpoint in ICE])
        return ( ret, UICE(ret))   
    
    ( (mincostplus, Uplus) , (mincostminus, Uminus) , (mincostICE, UICE) ) = (mincostplus(I, plus) ,  mincostminus(I, minus) , mincostICE(I, ICE))
    mincost = mincostplus + mincostminus + mincostICE
    alpha = conf.alpha/3.0
    gamma = conf.gamma
    cost = ((alpha *  gamma**(-mincostplus) )/ Uplus) + ((alpha *  gamma**(-mincostminus) )/ Uminus) + ((alpha *  gamma**(-mincostICE) )/ UICE)
    return (cost, mincost)

