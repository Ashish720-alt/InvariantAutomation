
""" Cost Functions.
This module includes cost functions.
"""

from z3 import *
import numpy as np
from repr import Repr
from dnf import DNF_to_z3expr, DNF_to_z3expr_p, op_norm_conj, norm_disj, norm_DNF, not_DNF, or_DNFs, and_DNFs
from configure import Configure as conf
from scipy.optimize import minimize, LinearConstraint


class Cost:
    def __init__(self, repr: Repr, I_arrs):
        self.num_var = repr.get_num_var()
        self.num_cex = conf.NUM_COUNTEREXAMPLE
        
        self.CHC = repr.CHC

        self.Is = {}
        self.Is_z3 = {}
        self.Ips_z3 = {}
        for j in range(len(repr.invs)):
            self.Is[repr.invs[j]] = I_arrs[j] 
            self.Is_z3[repr.invs[j]] = DNF_to_z3expr(I_arrs[j] )
            self.Ips_z3[repr.invs[j]] = DNF_to_z3expr_p(I_arrs[j] )
        
        self.DNFs = repr.DNFs
        self.DNFs_z3 = repr.DNFs_z3expr
        self.DNFps_z3 = repr.DNFs_p_z3expr

        self.trans_funcs = repr.trans_funcs
        self.trans_funcs_z3 = repr.trans_funcs_z3expr


        self.cex_list = {}
        for clause in self.CHC:
            self.cex_list[clause] = __get_cex( __get_z3_clause(clause) )

        self.cost_list = {} 
        # Get costFunction
        for clause in self.CHC:
            clause_type = self.__clause_type(clause)
            if (clause_type == -1):
                self.cost_list[clause] = self.__J_fact( self.cex_list[clause]  , clause)
            elif (clause_type == 0):
                self.cost_list[clause] = self.__J_inductive( self.cex_list[clause]  , clause) 
            else:
                self.cost_list[clause] = self.__J_query( self.cex_list[clause]  , clause)    
        
        self.cost = 0.0
        for clause in self.CHC:
            clause_type = self.__clause_type(clause)
            self.cost = self.cost + (conf.K[clause_type]*self.cost_list[clause])  

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
            # print(m)
            result.append(m)
            # Create a new constraint the blocks the current model
            block = []
            for d in m:
                # d is a declaration
                if d.arity() > 0:
                    raise Z3Exception(
                        "uninterpreted functions are not supported")
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

    # isarray = 1 if we want the numpy array, and 0 if we want z3 expression
    def __DNF_or_DNFz3(literal, primed, isarray):
        if (literal[0] == '~'):
            var = literal[1:]
            negation = 0
        else:
            var = literal
            negation = 1

        if (var in self.DNFs_z3.keys()):
            if (isarray == 1):
                ret = self.DNFs[var]
            else:
                ret = self.DNFps_z3[var] if (primed) else self.DNFs_z3[var]
        elif (var in self.Is_z3.keys()):
            if (isarray == 1):
                ret = self.Is[var]
            else:
                ret = self.Ips_z3[var] if (primed) else self.Is_z3[var]
        elif (var in self.trans_funcs_z3.keys()):
            if (isarray == 1):
                ret = self.trans_funcs [var]
            else:
                ret = self.trans_funcs_z3[var]
        else:
            if (conf.DISPLAY_WARNINGS == conf.ON):
                print("Error: Predicate form not allowed")
            return True
        if (isarray != 1):
            ret = Not(ret) if (negation == 1) else ret
        return ret


    def __get_z3_clause(self, clause):
        def f(c, primed):
            ret = True
            for p in c:
                ret = And( self.__DNF_or_DNFz3(p, primed, 0) , ret)
            return simplify(ret)

        LHS = clause[0]
        RHS = clause[1]
        is_transition = 1 if ( len( set(self.trans_funcs.keys()) & set(LHS) ) != 0) else 0

        return Implies( f(LHS, 0) , f(RHS, is_transition) )




    # Fact clause -> -1, Inductive clause -> 0, Query Clause -> 1
    def __clause_type(self, clause):
        LHS = clause[0]
        RHS = clause[1]

        ret = 0
        ret = ret - 1 if ( len( set(self.Is_z3.keys()) & set(LHS) ) != 0) else ret
        ret = ret + 1 if ( len( set(self.Is_z3.keys()) & set(RHS) ) != 0) else ret

        return ret

    """ Cost functions.
    """

    def __J_fact(self, cex_list, clause):
        RHS = clause[1]

        if ( len( set(self.trans_funcs.keys()) & set(RHS) ) != 0):
            if (DISPLAY_WARNINGS == ON):
                print("This fact clause has a transition function on the left:", clause)
            assert(( len( set(self.trans_funcs.keys()) & set(RHS) ) != 0))
        
        #for now, we assume that the RHS has only 1 symbol, which is a positive literal
            
        dnf = self.__DNF_or_DNFz3( RHS[0], 0, 1)
        
        for i in range(1, len(RHS)):
            dnf_temp = self.__DNF_or_DNFz3( RHS[i], 0, 1)
            dnf = and_DNFs( dnf, dnf_temp  )

        error = 0
        for cex in cex_list:
            # TODO: x%i is assumed across modules, should make it globally configured.
            pt = [cex.evaluate(Int("x%i"), model_completion=True).as_long()
                  for i in range(self.num_var)]
            point = np.array(pt)
            error = max(error, self.__distance_point_DNF(point, dnf))

        return error + len(cex_list)

    def __J_inductive(self, cex_list, clause):
        return len(cex_list)


    def __J_query(self, cex_list, clause):
        LHS = clause[0]

        if ( len( set(self.trans_funcs.keys()) & set(LHS) ) != 0):
            if (DISPLAY_WARNINGS == ON):
                print("This fact clause has a transition function on the left:", clause)
            assert(( len( set(self.trans_funcs.keys()) & set(LHS) ) != 0))
        
        #for now, we assume that the LHS has only 1 symbol, which is a positive literal
            
        dnf = self.__DNF_or_DNFz3( LHS[0], 0, 1)
        
        for i in range(1, len(LHS)):
            dnf_temp = self.__DNF_or_DNFz3( LHS[i], 0, 1)
            dnf = and_DNFs( dnf, dnf_temp  )

        error = 0
        for cex in cex_list:
            # TODO: x%i is assumed across modules, should make it globally configured.
            pt = [cex.evaluate(Int("x%i"), model_completion=True).as_long()
                  for i in range(self.num_var)]
            point = np.array(pt)
            error = max(error, self.__distance_point_DNF(point, dnf))

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
            C = op_norm_conj(C)
            A = np.concatenate(
                [C[:, :self.num_var], C[:, self.num_var+1:]], axis=1)
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

