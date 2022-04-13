
""" Cost Functions.
This module includes cost functions.
"""

from z3 import *
import numpy as np
from repr import Repr
from dnf import DNF_to_z3expr, DNF_to_z3expr_p, norm_conj
from configure import Configure as conf
from scipy.optimize import minimize, LinearConstraint

class Cost:
    def __init__(self, repr: Repr, I_arr: np.ndarray):
        self.num_var = repr.get_num_var()
        self.num_cex = conf.NUM_COUNTEREXAMPLE
        self.I = I_arr
        self.I_z3 = DNF_to_z3expr(I_arr)
        self.Ip_z3 = DNF_to_z3expr_p(I_arr)
        self.P = repr.get_P()
        self.B = repr.get_B()
        self.T = repr.get_T()
        self.Q = repr.get_Q()
        self.P_z3 = repr.get_P_z3expr()
        self.B_z3 = repr.get_B_z3expr()
        self.T_z3 = repr.get_T_z3expr()
        self.Q_z3 = repr.get_Q_z3expr()

        C1_cex_list = self.__get_cex_C1()
        C2_cex_list = self.__get_cex_C2()
        C3_cex_list = self.__get_cex_C3()

        # Get costFunction
        cost1 = self.__J1(C1_cex_list)
        cost2 = self.__J2(C2_cex_list)
        cost3 = self.__J3(C3_cex_list)
        self.cost = conf.K1*cost1 + conf.K2*cost2 + conf.K3*cost3

    def get_cost(self): 
        return self.cost

    """ Get counterexamples.
    """

    def __get_cex(self, C):
        result = []
        s = Solver()
        s.add(Not(C))
        while len(result) < self.num_cex and s.check() == sat:
            m = s.model()
            result.append(m)
            # Create a new constraint the blocks the current model
            block = []
            for d in m:
                # d is a declaration
                if d.arity() > 0:
                    raise Z3Exception("uninterpreted functions are not supported")
                # create a constant from declaration
                c = d()
                if is_array(c) or c.sort().kind() == Z3_UNINTERPRETED_SORT:
                    raise Z3Exception(
                        "arrays and uninterpreted sorts are not supported")
                block.append(c != m[d])
            s.add(Or(block))
        else:
            if len(result) < self.num_cex and s.check() != unsat:
                print("Solver can't verify or disprove")
                return result
        return result

    def __get_cex_C1(self): 
        """ p => I 
        """
        return self.__get_cex(Implies(self.P_z3, self.I_z3))

    def __get_cex_C2(self):
        """ B & I & T => I'
        """
        return self.__get_cex(Implies(And(self.B_z3, self.I_z3, self.T_z3), self.Ip_z3))

    def __get_cex_C3(self):
        """ I & !B => Q
        """
        return self.__get_cex(Implies(And(self.I_z3, Not(self.B_z3)), self.Q_z3))

    """ Cost functions.
    """
    
    def __J1(self, cex_list):
        error = 0
        for cex in cex_list:
            # TODO: x%i is assumed across modules, should make it globally configured.
            pt = [cex.evaluate(Int("x%i"), model_completion=True).as_long() 
                for i in range(self.num_var)]
            point = np.array(pt)
            error = max(error, self.__distance_point_DNF(point, self.I))
        return error + len(cex_list)

    # Traditionally try to 'guess' which cex are supposed to be negative, and which are supposed to be positive, and then there is a relative ratio; but we skip that here.


    def __J2(self, cex_list):
        return len(cex_list)


    def __J3(self, cex_list):
        error = 0
        for cex in cex_list:
            pt = [cex.evaluate(Int("x%i"), model_completion=True).as_long()
                for i in range(self.num_var)]
            point = np.array(pt)
            error = max(error, self.__distance_point_DNF(point, self.Q))
        return error + len(cex_list)


    """ Distance Functions.
    """

    def __distance_point_DNF(self, p, D: np.ndarray):
        """ 
        :point: an 1d array with len = num_var.
        :D: a DNF as np.ndarray.
        """
        assert(len(p) == self.num_var)

        def distance_point_conj_clauses(C):
            """
            :C: a 2d array, (conj, pred).
            """
            C = norm_conj(C)
            A = np.concatenate([C[:, :self.num_var], C[:, self.num_var+1:]], axis=1)
            return float(minimize(
                lambda x, p: np.linalg.norm(x - p),
                np.zeros(self.num_var),
                args=(p,),
                constraints=[LinearConstraint(A[:, :-1], -np.inf, -A[:, -1])],
            ).fun)

        d = float('inf')
        for C in D:
            d = min(d, distance_point_conj_clauses(C))
        return d

    # Testing:
    # print(distance_point_conj_clauses( np.array( [-3,1,3], ndmin = 1), np.array( [ [1,2,3,1,10], [1,3,1,0,0] ] , ndmin = 2)) )
    # print(distance_point_DNF( np.array( [-3,3,1], ndmin = 1), np.array( [ [ [-7,1,3,-2,3], [1,2,1,0,2], [3,1,3, 1, 4] ], [ [1,3,1,1,10], [1,3,1,0,0], [0,0,0,0,0] ] ]    )) )



