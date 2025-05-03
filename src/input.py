# Testing
from dnfs_and_transitions import dnfTrue, list3D_to_listof2Darrays, dnfdisjunction, dnfnegation
import numpy as np
from repr import genLItransitionrel, Repr

# -2: <, 2: >, 0: ==, -1: <=, 1: >=
# IMP: Remember Q is Q \/ B for standard CHC
# LARGE INT is 1000000


class Inputs:
    # class handcrafted:
    #     class mock:
    #         Var = ['x']
    #         P = [np.array([[1, 0, 0]])]
    #         B = [np.array([[1, -1, 5]])]
    #         Q = [np.array([[1, -1, 6]])]
    #         T = genLItransitionrel(
    #             B, ([np.array([[1, 1], [0, 1]])], dnfTrue(1)))
    #         c = 1
    #         d = 1
            

    #     class c2d1_1:
    #         Var = ['x', 'y']
    #         P = [np.array([[1, 0, 0, 1], [0, 1, 0, 1]])]
    #         B = dnfTrue(2)
    #         Q = [np.array([[1, -1, 1, 0], [0, 1, 1, 1]])]
    #         T = genLItransitionrel(
    #             B, ([np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])], dnfTrue(2)))
    #         c = 2  # c = 3 Value here shows larger c also converges
    #         d = 1
             

    class loop_lit:
        class afnp2014_modified:
            Var = ['x', 'y']
            P = [np.array([[1, 0, 0, 2], [0, 1, 0, 2]])]
            B = [np.array([[0, 1, -2, 1000]])]
            Q = [np.array([[1, -1, 1, 0]])]
            T = genLItransitionrel(
                B, ([np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])], dnfTrue(2)))
            c = 2
            d = 1
             

        class bhmr2007:
            Var = ['a', 'b', 'i', 'n']
            P = [np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [
                          0, 0, 0, 1, 1, 0], [0, 0, 0, 1, -1, 1000000]])]
            B = [np.array([[0, 0, 1, -1, -2, 0]])]
            Q = [np.array([[1, 1, 0, -3, 0, 0]]),
                 np.array([[0, 0, 1, -1, -1, 0]])]
            T = genLItransitionrel(B, ([np.array([[1, 0, 0, 0, 1], [0, 1, 0, 0, 2], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]),
                                        np.array([[1, 0, 0, 0, 2], [0, 1, 0, 0, 1], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])], dnfTrue(4)))
            c = 3
            d = 1
        class cggmp2005:
            Var = ['i', 'j']
            P = [np.array([[1, 0, 0, 1], [0, 1, 0, 10]])]
            B = [np.array([[-1, 1, 1, 0]])]
            Q = [np.array([[0, 1, 0, 6]]), np.array([[-1, 1, 1, 0]])]
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 2], [0, 1, -1], [0, 0, 1]])], dnfTrue(2)))
            c = 3
            d = 1
             

        class cggmp2005_variant:
            Var = ['lo', 'mid', 'hi']
            P = list3D_to_listof2Darrays(
                [[[1, 0, 0, 0, 0], [0, 1, 0, 2, 0], [0, 1, 0, -2, 1000000], [0, -2, 1, 0, 0]]])
            B = list3D_to_listof2Darrays([[[0, 1, 0, 2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[1, 0, -1, 0, 0]]]), B, 1)
            T = genLItransitionrel(B, ([np.array(
                [[1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1]])], dnfTrue(3)))
            c = 3
            d = 1
             

        class gsv2008:
            Var = ['x', 'y']
            P = [
                np.array([[1, 0, 0, -50], [0, 1, 2, 1000], [0, 1, -2, 1000000]])]
            B = [np.array([[1, 0, -2, 0]])]
            Q = [np.array([[0, 1, 2, 0]]), np.array([[1, 0, -2, 0]])]
            T = genLItransitionrel(
                B, ([np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])], dnfTrue(2)))
            c = 1
            d = 2
             

        class css2003:
            Var = ['i', 'j', 'k']
            P = list3D_to_listof2Darrays(
                [[[1, 0, 0, 0, 1], [0, 1, 0, 0, 1], [0, 0, 1, 1, 0], [0, 0, 1, -1, 1]]])
            B = list3D_to_listof2Darrays([[[1, 0, 0, -2, 1000000]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[1, 0, 1, -1, 2], [1, 0, 0, 1, 1], [-1, 0, -1, -1, -1]]]), dnfnegation(B), 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 1, -1], [0, 0, 0, 1]])], dnfTrue(3)))
            c = 3
            d = 1
             
             

        class gj2007:
            Var = ['x', 'y']
            P = list3D_to_listof2Darrays([[[1, 0, 0, 0], [0, 1, 0, 50]]])
            B = list3D_to_listof2Darrays([[[1, 0, -2, 100]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[0, 1, 0, 100]]]), B, 1)
            T = genLItransitionrel(B, ([np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])], list3D_to_listof2Darrays([[[1, 0, -2, 50]]])),
                                   ([np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])],
                                    list3D_to_listof2Darrays([[[1, 0, 1, 50]]]))
                                   )
            c = 5
            d = 2
            clist = [4,1]

        class gj2007b:
            Var = ['x', 'm', 'n']
            P = list3D_to_listof2Darrays(
                [[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0],  [0, 0, 1, 2, 0], [0,0,1,2,0]]])
            B = list3D_to_listof2Darrays([[[1, 0, -1, -2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[0, 1, 0, 1, 0], [0, 1, -1, -2, 0]] ]), B, 1)
            T = genLItransitionrel(B, ([np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                                        np.array([[1, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])], dnfTrue(3)))
            c = 4
            d = 2
            clist = [4,2]

        class gr2006:
            Var = ['x', 'y']
            P = list3D_to_listof2Darrays([[[1, 0, 0, 0], [0, 1, 0, 0]]])
            B = list3D_to_listof2Darrays(
                [[[1, 0, -2, 50], [0, 1, 1, -1]], [[1, 0, 1, 50], [0, 1, 1, 1]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[1, 0, 0, 100]]]), B, 1)
            T = genLItransitionrel(B, ([np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]]),], list3D_to_listof2Darrays([[[1, 0, -2, 50]]])),
                                   ([np.array([[1, 0, 1], [0, 1, -1], [0, 0, 1]]),],
                                    list3D_to_listof2Darrays([[[1, 0, 1, 50]]]))
                                   )
            c = 4
            d = 2
            clist = [4,3]
            
        class hhk2008:
            Var = ['a', 'b', 'res', 'cnt']
            P = list3D_to_listof2Darrays([[[-1, 0, 1, 0, 0, 0], [0, -1, 0, 1, 0, 0], [
                                         1, 0, 0, 0, -1, 1000000], [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, -1, 1000000]]])
            B = list3D_to_listof2Darrays([[[0, 0, 0, 1, 2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[-1, -1, 1, 0, 0, 0]]]), B, 1)
            T = genLItransitionrel(B, ([np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [
                                   0, 0, 1, 0, 1], [0, 0, 0, 1, -1], [0, 0, 0, 0, 1]])], dnfTrue(4)))
            c = 3
            d = 1
             

        class jm2006:
            Var = ['i', 'j', 'x', 'y']
            P = list3D_to_listof2Darrays(
                [[[1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0], [-1, 0, 1, 0, 0, 0], [0, -1, 0, 1, 0, 0]]])
            B = list3D_to_listof2Darrays(
                [[[0, 0, 1, 0, 2, 0]], [[0, 0, 1, 0, -2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[1, -1, 0, 0, -2, 0]], [[1, -1, 0, 0, 2, 0]], [[0, 0, 0, 1, 0, 0]]]), B, 1)
            T = genLItransitionrel(B, ([np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [
                                   0, 0, 1, 0, -1], [0, 0, 0, 1, -1], [0, 0, 0, 0, 1]])], dnfTrue(4)))
            c = 2
            d = 1
             

        class jm2006_variant:
            Var = ['i', 'j', 'x', 'y', 'z']
            P = list3D_to_listof2Darrays([[[1, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, -1, 1000000], [
                                         0, 1, 0, 0, 0, 1, 0], [-1, 0, 1, 0, 0, 0, 0], [0, -1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]]])
            B = list3D_to_listof2Darrays(
                [[[0, 0, 1, 0, 0, 2, 0]], [[0, 0, 1, 0, 0, -2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[1, -1, 0, 0, 0, -2, 0]], [[1, -1, 0, 0, 0, 2, 0]], [[0, 0, 0, 1, 1, 0, 0]]]), B, 1)
            T = genLItransitionrel(B, ([np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [
                                   0, 0, 1, 0, 0, -1], [0, 0, 0, 1, 0, -2], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 1]])], dnfTrue(5)))
            c = 2
            d = 1
             

    class loop_new:

        class count_by_1:
            Var = ['i']
            P = list3D_to_listof2Darrays([[[1, 0, 0]]])
            B = list3D_to_listof2Darrays([[[1, -2, 1000000]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[1, 0, 1000000]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 1], [0, 1]])], dnfTrue(1)))
            c = 1
            d = 1
        class count_by_1_variant:
            Var = ['i']
            P = list3D_to_listof2Darrays([[[1, 0, 0]]])
            B = list3D_to_listof2Darrays([[[1, -2, 1000000]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[1, -1, 1000000]]]), dnfnegation(B), 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 1], [0, 1]])], dnfTrue(1)))
            c = 1
            d = 1
             
             

    class loop_simple:
        class nested_1:
            Var = ['i']
            P = [np.array([[1, 0, 0]])]
            B = [np.array([[1, -2, 6]])]
            Q = [np.array([[1, -1, 6]])]
            T = genLItransitionrel(
                B, ([np.array([[1, 1], [0, 1]])], dnfTrue(1)))
            c = 1
            d = 1
             

    class loop_zilu:
        class benchmark01_conjunctive:
            Var = ['x', 'y']
            P = list3D_to_listof2Darrays([[[1, 0, 0, 1], [0, 1, 0, 1]]])
            B = dnfTrue(2)
            Q = list3D_to_listof2Darrays([[[0, 1, 1, 1]]])
            T = genLItransitionrel(
                B, ([np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])], dnfTrue(2)))
            c = 3
            d = 1
             

        class benchmark02_linear:
            Var = ['n', 'i', 'l']
            P = list3D_to_listof2Darrays([[[0, 0, 1, 2, 0], [0, 1, -1, 0, 0]]])
            B = list3D_to_listof2Darrays([[[-1, 1, 0, -2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[0, 0, 1, 1, 1]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])], dnfTrue(3)))
            c = 1
            d = 1
             

        class benchmark03_linear_modified:
            Var = ['x', 'y', 'i', 'j']
            P = list3D_to_listof2Darrays([[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [
                                         0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]]])
            B = dnfTrue(4)
            Q = list3D_to_listof2Darrays([[[0, 0, -1, 1, 1, 0]]])
            T = genLItransitionrel(B, ([np.array([[1, 0, 0, 0, 1], [0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 1]]),
                                        np.array([[1, 0, 0, 0, 1], [0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 1], [0, 0, 0, 0, 1]])], dnfTrue(4)))
            c = 2
            d = 1
             

        class benchmark04_conjunctive:
            Var = ['k', 'j', 'n']
            P = list3D_to_listof2Darrays(
                [[[0, 0, 1, 1, 1], [1, 0, -1, 1, 0], [0, 1, 0, 0, 0]]])
            B = list3D_to_listof2Darrays([[[0, 1, -1, -1, -1]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[1, 0, 0, 1, 0]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 0, -1], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])], dnfTrue(3)))
            c = 2
            d = 1
             

        # correct version!
        class benchmark05_conjunctive:
            Var =  ['x', 'y', 'n']
            P = list3D_to_listof2Darrays([[[1, 0, 0, 1, 0], [1, -1, 0, -1, 0], [0, 1, -1, -2, 0]]])
            B = list3D_to_listof2Darrays([[[1, 0, -1, -2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 1, -1, 0, 0]]]) , B , 1)
            T = genLItransitionrel(B,
                                ( [ np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) ]
                                    , list3D_to_listof2Darrays([[[1, -1, 0, -2, 0]]]) ),
                                ( [ np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]) ]
                                    , list3D_to_listof2Darrays([[[1, -1, 0, 1, 0]]]) ),
                                        )
            c = 4
            d = 2

        class benchmark05_conjunctive:
            Var = ['x', 'y', 'n']
            P = list3D_to_listof2Darrays(
                [[[1, 0, 0, 1, 0], [1, -1, 0, -1, 0], [0, 1, -1, -2, 0]]])
            B = list3D_to_listof2Darrays([[[1, 0, -1, -2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[0, 1, -1, 0, 0]]]), B, 1)
            T = genLItransitionrel(B,
                                   ([np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0], [
                                    0, 0, 0, 1]])], list3D_to_listof2Darrays([[[1, -1, 0, 2, -1]]])),
                                   ([np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [
                                    0, 0, 0, 1]])], list3D_to_listof2Darrays([[[1, -1, 0, -1, -1]]])),
                                   )
            c = 2
            d = 1
             

        class benchmark06_conjunctive:
            Var = ['i', 'j', 'x', 'y', 'k']
            P = list3D_to_listof2Darrays(
                [[[0, 0, 1, 1, -1, 0, 0], [0, 1, 0, 0, 0, 0, 0]]])
            B = dnfTrue(5)
            Q = list3D_to_listof2Darrays([[[0, 0, 1, 1, -1, 0, 0]]])
            T = genLItransitionrel(B,
                                   ([np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, -1], [0, 0, 0, 0, 1, 0],
                                               [0, 0, 0, 0, 0, 1]])], list3D_to_listof2Darrays([[[-1, 1, 0, 0, 0, 0, 0]]])),
                                   ([np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, -1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 0],
                                               [0, 0, 0, 0, 0, 1]])], list3D_to_listof2Darrays([[[-1, 1, 0, 0, 0, -2, 0]], [[1, -1, 0, 0, 0, -2, 0]]])))
            c = 2
            d = 1
             
             

        class benchmark08_conjunctive:
            Var = ['n', 'sum', 'i']
            P = list3D_to_listof2Darrays(
                [[[1, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]])
            B = list3D_to_listof2Darrays([[[-1, 0, 1, -2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[0, 1, 0, 1, 0]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]])], dnfTrue(3)))
            c = 2
            d = 1

        class benchmark09_conjunctive:
            Var = ['x', 'y']
            P = list3D_to_listof2Darrays([[[1, -1, 0, 0], [0, 1, 1, 0]]])
            B = list3D_to_listof2Darrays([[[0, 1, -2, 0]], [[0, 1, 2, 0]]])
            Q = dnfdisjunction(
                list3D_to_listof2Darrays([[[0, 1, 0, 0]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, -1], [0, 1, -1], [0, 0, 1]])], dnfTrue(2)))
            c = 1
            d = 1
             

        class benchmark10_conjunctive:
            Var = ['i', 'c']
            P = list3D_to_listof2Darrays([[[0, 1, 0, 0], [1, 0, 0, 0]]])
            B = list3D_to_listof2Darrays([[[1, 0, -2, 100], [1, 0, 2, -1]]])
            Q = dnfdisjunction(
                list3D_to_listof2Darrays([[[0, 1, 1, 0]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1]])], dnfTrue(2)))
            c = 2
            d = 1
             

        class benchmark11_linear:
            Var = ['x', 'n']
            P = list3D_to_listof2Darrays([[[1, 0, 0, 0], [0, 1, 2, 0]]])
            B = list3D_to_listof2Darrays([[[1, -1, -2, 0]]])
            Q = dnfdisjunction(
                list3D_to_listof2Darrays([[[1, -1, 0, 0]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])], dnfTrue(2)))
            c = 1
            d = 1
             

        class benchmark12_linear:
            Var = ['x', 'y', 't']
            P = list3D_to_listof2Darrays(
                [[[1, -1, 0, 2, 0], [0, 1, -1, 0, 0]], [[1, -1, 0, -2, 0], [0, 1, -1, 0, 0]]])
            B = dnfTrue(3)
            Q = list3D_to_listof2Darrays([[[0, 1, -1, 1, 0]]])
            T = genLItransitionrel(B,
                                   ([np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [
                                    0, 0, 0, 1]])], list3D_to_listof2Darrays([[[1, 0, 0, 2, 0]]])),
                                   ([np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [
                                    0, 0, 0, 1]])], list3D_to_listof2Darrays([[[1, 0, 0, -1, 0]]]))
                                   )
            c = 1
            d = 1
             

        class benchmark13_conjunctive:
            Var = ['i', 'j', 'k']
            P = list3D_to_listof2Darrays([[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]])
            B = list3D_to_listof2Darrays([[[1, 0, -1, -1, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[-1, 1, 0, 0, 0]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])], dnfTrue(3)))
            c = 2
            d = 1
             

        class benchmark14_linear:
            Var = ['i']
            P = list3D_to_listof2Darrays([[[1, 1, 0], [1, -1, 200]]])
            B = list3D_to_listof2Darrays([[[1, 2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, 1, 0]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, -1], [0, 1]])], dnfTrue(1)))
            c = 2
            d = 1
             

        class benchmark15_conjunctive:
            Var = ['low', 'mid', 'high']
            P = list3D_to_listof2Darrays(
                [[[1, 0, 0, 0, 0], [0, 1, 0, 1, 1], [0, -2, 1, 0, 0]]])
            B = list3D_to_listof2Darrays([[[0, 1, 0, 2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[1, 0, -1, 0, 0]]]), B, 1)
            T = genLItransitionrel(B, ([np.array(
                [[1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1]])], dnfTrue(3)))
            c = 3
            d = 1
             

        class benchmark16_conjunctive:
            Var = ['i', 'k']
            P = list3D_to_listof2Darrays(
                [[[0, -1, -1, 0], [0, 1, -1, 1], [1, 0, 0, 1]]])
            B = dnfTrue(2)
            Q = list3D_to_listof2Darrays(
                [[[-1, -1, -1, -1], [1, 1, -1, 2], [1, 0, 1, 1]]])
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 1], [0, 1, -1], [0, 0, 1]])], dnfTrue(2)))
            c = 3
            d = 1
             

        class benchmark17_conjunctive:
            Var = ['i', 'k', 'n']
            P = list3D_to_listof2Darrays([[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]])
            B = list3D_to_listof2Darrays([[[1, 0, -1, -2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[0, 1, -1, 1, 0]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])], dnfTrue(3)))
            c = 2
            d = 1
            
        class benchmark18_conjunctive:
            Var = ['i', 'k', 'n']
            P = list3D_to_listof2Darrays(
                [[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 2, 0]]])
            B = list3D_to_listof2Darrays([[[1, 0, -1, -2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[0, 1, -1, 0, 0], [1, -1, 0, 0, 0]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])], dnfTrue(3)))
            c = 3
            d = 1
             

        class benchmark19_conjunctive:
            Var = ['j', 'k', 'n']
            P = list3D_to_listof2Darrays(
                [[[1, 0, -1, 0, 0], [0, 1, -1, 0, 0], [0, 0, 1, 2, 0]]])
            B = list3D_to_listof2Darrays([[[0, 0, 1, 2, 0], [1, 0, 0, 2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[0, 1, 0, 0, 0], [0, 1, 0, 0, 0]]]), B, 1)
            T = genLItransitionrel(B, ([np.array(
                [[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, 0], [0, 0, 0, 1]])], dnfTrue(3)))
            c = 4
            d = 1
             

        class benchmark20_conjunctive:
            Var = ['i', 'n', 'sum']
            P = list3D_to_listof2Darrays(
                [[[1, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 1, 0, -1, 100], [0, 0, 1, 0, 0]]])
            B = list3D_to_listof2Darrays([[[1, -1, 0, -2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[0, 0, 1, 1, 0]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]])], dnfTrue(3)))
            c = 2
            d = 1
             

        class benchmark21_disjunctive:
            Var = ['x', 'y']
            P = list3D_to_listof2Darrays([[[0, 1, 2, 0]], [[1, 0, 2, 0]]])
            B = list3D_to_listof2Darrays([[[1, 1, -1, -2]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[0, 1, 2, 0]], [[1, 0, 2, 0]]]), B, 1)
            T = genLItransitionrel(B, ([np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])], list3D_to_listof2Darrays([[[1, 0, 2, 0]]])),
                                   ([np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])],
                                    list3D_to_listof2Darrays([[[1, 0, -1, 0]]])),
                                   )
            c = 1
            d = 2
        
        class benchmark22_conjunctive:
            Var = ['x', 'y']
            P = list3D_to_listof2Darrays([[[1, 0, 0, 1], [0, 1, 0, 0]]])
            B = dnfTrue(2)
            Q = list3D_to_listof2Darrays([[[1, -1, 1, 0]]])
            T = genLItransitionrel(
                B, ([np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])], dnfTrue(2)))
            c = 3
            d = 1
             

        class benchmark23_conjunctive:
            Var = ['i', 'j']
            P = list3D_to_listof2Darrays([[[1, 0, 0, 0], [0, 1, 0, 0]]])
            B = list3D_to_listof2Darrays([[[1, 0, -2, 100]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[0, 1, 0, 200]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 1], [0, 1, 2], [0, 0, 1]])], dnfTrue(2)))
            c = 3
            d = 1
        
        class benchmark24_conjunctive:
            Var = ['i', 'k', 'n']
            P = list3D_to_listof2Darrays(
                [[[1, 0, 0, 0, 0], [0, 1, -1, 0, 0], [0, 0, 1, 1, 0]]])
            B = list3D_to_listof2Darrays([[[1, 0, -1, -2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[0, 2, -1, 1, -1]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 0, 2], [0, 1, 0, -1], [0, 0, 1, 0], [0, 0, 0, 1]])], dnfTrue(3)))
            c = 2
            d = 1
             

        class benchmark25_linear:
            Var = ['x']
            P = list3D_to_listof2Darrays([[[1, -2, 0]]])
            B = list3D_to_listof2Darrays([[[1, -2, 10]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, 0, 10]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 1], [0, 1]])], dnfTrue(1)))
            c = 1
            d = 1
        
        class benchmark26_linear:
            Var = ['x', 'y']
            P = list3D_to_listof2Darrays([[[1, -1, -2, 0]]])
            B = list3D_to_listof2Darrays([[[1, -1, -2, 0]]])
            Q = dnfdisjunction(
                list3D_to_listof2Darrays([[[1, -1, 0, 0]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])], dnfTrue(2)))
            c = 1
            d = 1
             

        class benchmark27_linear:
            Var = ['i', 'j', 'k']
            P = list3D_to_listof2Darrays(
                [[[1, -1, 0, -2, 0], [-1, 1, 1, 2, 0]]])
            B = list3D_to_listof2Darrays([[[1, -1, 0, -2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[0, 0, 1, 2, 0]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])], dnfTrue(3)))
            c = 1
            d = 1
             

        class benchmark28_linear:
            Var = ['i', 'j']
            P = dnfdisjunction(list3D_to_listof2Darrays([[[1, -1, -2, 0], [1, 1, 2, 0], [0, 1, 1, 0]]]),
                               list3D_to_listof2Darrays([[[1, -1, 2, 0], [1, 1, -2, 0], [0, 1, 1, 0]]]), 1)
            B = list3D_to_listof2Darrays([[[1, -1, -2, 0]]])
            Q = dnfdisjunction(
                list3D_to_listof2Darrays([[[-1, 1, 0, 0]]]), B, 1)
            T = genLItransitionrel(B, ([np.array([[-1, 1, 0], [1, 0, 0], [0, 0, 1]])], list3D_to_listof2Darrays([[[-2, 1, -2, 0]]])),
                                   ([np.array([[1, 0, 0], [-1, 1, 0], [0, 0, 1]])], list3D_to_listof2Darrays([[[-2, 1, 1, 0]]])))
            c = 1
            d = 1
             

        class benchmark29_linear:
            Var = ['x', 'y']
            P = list3D_to_listof2Darrays([[[1, -1, -2, 0]]])
            B = list3D_to_listof2Darrays([[[1, -1, -2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[1, -1, 1, 0], [1, -1, -1, 99]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 100], [0, 1, 0], [0, 0, 1]])], dnfTrue(2)))
            c = 1
            d = 1
             

        class benchmark30_conjunctive:
            Var = ['x', 'y']
            P = list3D_to_listof2Darrays([[[-1, 1, 0, 0]]])
            B = dnfTrue(2)
            Q = list3D_to_listof2Darrays([[[1, -1, 0, 0]]])
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])], dnfTrue(2)))
            c = 2
            d = 1
             

        class benchmark31_disjunctive:
            Var = ['x', 'y']
            P = list3D_to_listof2Darrays([[[1, 0, -2, 0]]])
            B = list3D_to_listof2Darrays([[[1, 0, -2, 0]]])
            Q = dnfdisjunction(
                list3D_to_listof2Darrays([[[0, 1, 1, 0]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])], dnfTrue(2)))
            c = 1
            d = 3
             

        class benchmark32_linear:
            Var = ['x']
            P = list3D_to_listof2Darrays([[[1, 0, 1]], [[1, 0, 2]]])
            B = dnfTrue(1)
            Q = list3D_to_listof2Darrays([[[1, -1, 8]]])
            T = genLItransitionrel(B, ([np.array([[0, 2], [0, 1]])], list3D_to_listof2Darrays([[[1, 0, 1]]])),
                                   ([np.array([[0, 1], [0, 1]])],
                                    list3D_to_listof2Darrays([[[1, 0, 2]]])),
                                   ([np.array([[1, 0], [0, 1]])], list3D_to_listof2Darrays(
                                       [[[1, -1, 0]], [[1, 1, 3]]]))
                                   )
            c = 1
            d = 1
             

        class benchmark33_linear:
            Var = ['x']
            P = list3D_to_listof2Darrays([[[1, 1, 0]]])
            B = list3D_to_listof2Darrays([[[1, -2, 100], [1, 1, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, 1, 100]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 1], [0, 1]])], dnfTrue(1)))
            c = 1
            d = 1
             

        class benchmark34_conjunctive:
            Var = ['j', 'k', 'n']
            P = list3D_to_listof2Darrays(
                [[[1, 0, 0, 0, 0], [0, 1, -1, 0, 0], [0, 0, 1, 2, 0]]])
            B = list3D_to_listof2Darrays(
                [[[0, 0, 1, 2, 0], [1, 0, -1, -2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[0, 1, 0, 0, 0], [0, 1, 0, -1, 1]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 0], [0, 0, 0, 1]])], dnfTrue(3)))
            c = 4
            d = 1
             

        class benchmark35_linear:
            Var = ['x']
            P = list3D_to_listof2Darrays([[[1, 1, 0]]])
            B = list3D_to_listof2Darrays([[[1, 1, 0], [1, -2, 10]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, 1, 10]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 1], [0, 1]])], dnfTrue(1)))
            c = 1
            d = 1
             

        class benchmark36_conjunctive:
            Var = ['x', 'y']
            P = list3D_to_listof2Darrays([[[1, -1, 0, 0], [0, 1, 0, 0]]])
            B = dnfTrue(2)
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[1, -1, 0, 0], [1, 0, 1, 0]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])], dnfTrue(2)))
            c = 3
            d = 1
             

        class benchmark37_conjunctive:
            Var = ['x', 'y']
            P = list3D_to_listof2Darrays([[[1, -1, 0, 0], [1, 0, 1, 0]]])
            B = list3D_to_listof2Darrays([[[1, 0, 2, 0]]])
            Q = dnfdisjunction(
                list3D_to_listof2Darrays([[[0, 1, 1, 0]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, -1], [0, 1, -1], [0, 0, 1]])], dnfTrue(2)))
            c = 3
            d = 1
            
        class benchmark38_conjunctive:
            Var = ['x', 'y']
            P = list3D_to_listof2Darrays([[[1, -1, 0, 0], [0, 1, 0, 0]]])
            B = dnfTrue(2)
            Q = list3D_to_listof2Darrays([[[1, -4, 0, 0], [1, 0, 1, 0]]])
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 4], [0, 1, 1], [0, 0, 1]])], dnfTrue(2)))
            c = 3
            d = 1
            
        class benchmark39_conjunctive:
            Var = ['x', 'y']
            P = list3D_to_listof2Darrays([[[1, -4, 0, 0], [1, 0, 1, 0]]])
            B = list3D_to_listof2Darrays([[[1, 0, 2, 0]]])
            Q = dnfdisjunction(
                list3D_to_listof2Darrays([[[0, 1, 1, 0]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, -4], [0, 1, -1], [0, 0, 1]])], dnfTrue(2)))
            c = 3
            d = 1
            
        class benchmark40_polynomial:
            Var = ['x', 'y']
            P = list3D_to_listof2Darrays([[[1, 0, -1, 0], [0, 1, -1, 0]]])
            B = dnfTrue(2)
            Q = list3D_to_listof2Darrays([[[1, 0, -1, 0], [0, 1, -1, 0]]])
            T = genLItransitionrel(B, ([np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])], list3D_to_listof2Darrays([[[1, 0, 2, 0]]])),
                                   ([np.array([[1, 0, -1], [0, 1, 0], [0, 0, 1]])],
                                    list3D_to_listof2Darrays([[[1, 0, -2, 0]]])),
                                   ([np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])], list3D_to_listof2Darrays(
                                       [[[1, 0, 0, 0], [0, 1, 2, 0]]])),
                                   ([np.array([[1, 0, -1], [0, 1, 0], [0, 0, 1]])],
                                    list3D_to_listof2Darrays([[[1, 0, 0, 0], [0, 1, -1, 0]]]))
                                   )
            c = 2
            d = 2
            
        class benchmark41_conjunctive:
            Var = ['x', 'y', 'z']
            P = list3D_to_listof2Darrays(
                [[[1, -1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]])
            B = dnfTrue(3)
            Q = list3D_to_listof2Darrays(
                [[[1, -1, 0, 0, 0], [1, 0, 0, 1, 0], [1, 1, 1, 0, 0]]])
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, -2], [0, 0, 0, 1]])], dnfTrue(3)))
            c = 5
            d = 1
            
        class benchmark42_conjunctive:
            Var = ['x', 'y', 'z']
            P = list3D_to_listof2Darrays(
                [[[1, -1, 0, 0, 0], [1, 0, 0, 1, 0], [1, 1, 1, 0, 0]]])
            B = list3D_to_listof2Darrays([[[1, 0, 0, 2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[0, 0, 1, -1, 0]]]), B, 1)
            T = genLItransitionrel(B, ([np.array(
                [[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, 2], [0, 0, 0, 1]])], dnfTrue(3)))
            c = 5
            d = 1

        class benchmark43_conjunctive:
            Var = ['x', 'y']
            P = list3D_to_listof2Darrays([[[1, 0, -2, 100], [0, 1, -2, 100]]])
            B = list3D_to_listof2Darrays([[[1, 0, -2, 100], [0, 1, -2, 100]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[1, 0, 0, 100]], [[0, 1, 0, 100]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])], dnfTrue(2)))
            c = 2
            d = 1

        class benchmark44_disjunctive:
            Var = ['x', 'y']
            P = list3D_to_listof2Darrays([[[1, -1, -2, 0]]])
            B = list3D_to_listof2Darrays([[[1, -1, -2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[1, -1, 1, 0], [1, -1, -1, 16]]]), B, 1)
            T = genLItransitionrel(B, ([np.array([[1, 0, 7], [0, 1, -10], [0, 0, 1]])], list3D_to_listof2Darrays([[[1, 0, -2, 0], [0, 1, -2, 0]]])),
                                   ([np.array([[1, 0, 7], [0, 1, 3], [0, 0, 1]])], list3D_to_listof2Darrays(
                                       [[[1, 0, -2, 0], [0, 1, 1, 0]]])),
                                   ([np.array([[1, 0, 10], [0, 1, 3], [0, 0, 1]])],
                                    list3D_to_listof2Darrays([[[1, 0, 1, 0]]])),
                                   )
            c = 1
            d = 1

        class benchmark45_disjunctive:
            Var = ['x', 'y']
            P = list3D_to_listof2Darrays([[[0, 1, 2, 0]], [[1, 0, 2, 0]]])
            B = dnfTrue(2)
            Q = list3D_to_listof2Darrays([[[1, 0, 2, 0]], [[0, 1, 2, 0]]])
            T = genLItransitionrel(B, ([np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])], list3D_to_listof2Darrays([[[1, 0, 2, 0]]])),
                                   ([np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])],
                                    list3D_to_listof2Darrays([[[1, 0, -1, 0]]]))
                                   )
            c = 1
            d = 2

        class benchmark46_disjunctive:
            Var = ['x', 'y', 'z']
            P = list3D_to_listof2Darrays(
                [[[0, 1, 0, 2, 0]], [[1, 0, 0, 2, 0]], [[0, 0, 1, 2, 0]]])
            B = dnfTrue(3)
            Q = list3D_to_listof2Darrays(
                [[[0, 1, 0, 2, 0]], [[1, 0, 0, 2, 0]], [[0, 0, 1, 2, 0]]])
            T = genLItransitionrel(B, ([np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])], list3D_to_listof2Darrays([[[1, 0, 0, 2, 0], [0, 1, 0, 2, 0]]])),
                                   ([np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 1], [
                                    0, 0, 0, 1]])], list3D_to_listof2Darrays([[[1, 0, 0, 2, 0], [0, 1, 0, -1, 0]]])),
                                   ([np.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0], [
                                    0, 0, 0, 1]])], list3D_to_listof2Darrays([[[1, 0, 0, -1, 0], [0, 1, 0, 2, 0]]])),
                                   ([np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [
                                    0, 0, 0, 1]])], list3D_to_listof2Darrays([[[1, 0, 0, -1, 0], [0, 1, 0, -1, 0]]]))
                                   )
            c = 1
            d = 3
        class benchmark47_linear:
            Var = ['x', 'y']
            P = list3D_to_listof2Darrays([[[1, -1, -2, 0]]])
            B = list3D_to_listof2Darrays([[[1, -1, -2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[1, -1, 1, 0], [1, -1, -1, 16]]]), B, 1)
            T = genLItransitionrel(B, ([np.array([[1, 0, 7], [0, 1, -10], [0, 0, 1]])], list3D_to_listof2Darrays([[[1, 0, -2, 0], [0, 1, -2, 0]]])),
                                   ([np.array([[1, 0, 10], [0, 1, -10], [0, 0, 1]])],
                                    list3D_to_listof2Darrays([[[1, 0, 1, 0], [0, 1, -2, 0]]])),
                                   ([np.array([[1, 0, 7], [0, 1, 3], [0, 0, 1]])], list3D_to_listof2Darrays(
                                       [[[1, 0, -2, 0], [0, 1, 1, 0]]])),
                                   ([np.array([[1, 0, 10], [0, 1, 3], [0, 0, 1]])], list3D_to_listof2Darrays(
                                       [[[1, 0, 1, 0], [0, 1, 1, 0]]]))
                                   )
            c = 1
            d = 1

        class benchmark48_linear:
            Var = ['i', 'j', 'k']
            P = list3D_to_listof2Darrays(
                [[[1, -1, 0, -2, 0], [0, 0, 1, 2, 0]]])
            B = list3D_to_listof2Darrays([[[1, -1, 0, -2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[1, -1, 1, 2, 0]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])], dnfTrue(3)))
            c = 1
            d = 1
             
            
        
        class benchmark49_linear:
            Var = ['i', 'j', 'r']
            P = list3D_to_listof2Darrays([[[-1, -1, 1, 2, 0]]])
            B = list3D_to_listof2Darrays([[[1, 0, 0, 2, 0]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays(
                [[[-1, -1, 1, 2, 0]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, 0, -1], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])], dnfTrue(3)))
            c = 1
            d = 1

        class benchmark50_linear:
            Var = ['xa', 'ya']
            P = list3D_to_listof2Darrays([[[1, 1, 2, 0]]])
            B = list3D_to_listof2Darrays([[[1, 0, 2, 0]]])
            Q = dnfdisjunction(
                list3D_to_listof2Darrays([[[0, 1, 1, 0]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 0, -1], [0, 1, 1], [0, 0, 1]])], dnfTrue(2)))
            c = 1
            d = 1
        class benchmark51_polynomial:
            Var = ['x']
            P = list3D_to_listof2Darrays([[[1, 1, 0], [1, -1, 50]]])
            B = dnfTrue(1)
            Q = list3D_to_listof2Darrays([[[1, 1, 0], [1, -1, 50]]])
            T = genLItransitionrel(B, ([np.array([[1, 1], [0, 1]])], list3D_to_listof2Darrays([[[1, 0, 0]], [[1, 2, 50]]])),
                                   ([np.array([[1, -1], [0, 1]])], list3D_to_listof2Darrays(
                                       [[[1, -2, 0]], [[1, 2, 0], [1, -1, 50]]]))
                                   )
            c = 2
            d = 1
        class benchmark52_polynomial:
            Var = ['i']
            P = list3D_to_listof2Darrays([[[1, -2, 10], [1, 2, -10]]])
            B = list3D_to_listof2Darrays([[[1, -2, 10], [1, 2, -10]]])
            Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, 0, 10]]]), B, 1)
            T = genLItransitionrel(
                B, ([np.array([[1, 1], [0, 1]])], dnfTrue(1)))
            c = 2
            d = 1
        class benchmark53_polynomial:
            Var = ['x', 'y']
            P = list3D_to_listof2Darrays([[[1, 0, -1, 0], [0, 1, -1, 0]]])
            B = dnfTrue(2)
            Q = list3D_to_listof2Darrays([[[1, 0, -1, 0], [0, 1, -1, 0]]])
            T = genLItransitionrel(B, ([np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])], list3D_to_listof2Darrays([[[1, 0, 2, 0]]])),
                                   ([np.array([[1, 0, -1], [0, 1, 0], [0, 0, 1]])],
                                    list3D_to_listof2Darrays([[[1, 0, -2, 0]]])),
                                   ([np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])], list3D_to_listof2Darrays(
                                       [[[1, 0, 0, 0], [0, 1, 2, 0]]])),
                                   ([np.array([[1, 0, -1], [0, 1, 0], [0, 0, 1]])],
                                    list3D_to_listof2Darrays([[[1, 0, 0, 0], [0, 1, -1, 0]]]))
                                   )
            c = 2
            d = 2


#     class breakGSpacer:
#         class b:
#             Var = ['x', 'y']
#             P = [np.array([[1, 0, 0, 1] , [0, 1, 0, 0]])]
#             B = dnfTrue(2)
#             Q = [np.array([[1, -1, 1, 0]])]
#             T = genLItransitionrel(
#                 B, ([np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])], dnfTrue(2)))
#             c = 2
#             d = 1
#             #  

#         class b1:
#             Var = ['x', 'y']
#             P = [np.array([[1, 0, 0, 1] , [0, 1, 0, 0]])]
#             B = dnfTrue(2)
#             Q = [np.array([[1, -1, 1, 0]])]
#             T = genLItransitionrel(
#                 B, ([np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]])], dnfTrue(2)))
#             c = 2
#             d = 1
#             #  

#         class b2:
#             Var = ['x', 'y']
#             P = [np.array([[1, 0, 0, 1] , [0, 1, 0, 0]])]
#             B = dnfTrue(2)
#             Q = [np.array([[1, -1, 1, 0]])]
#             T = genLItransitionrel(
#                 B, ([np.array([[2, 1, 0], [0, 1, 0], [0, 0, 1]])], dnfTrue(2)))
#             c = 2
#             d = 1

#         class b3:
#             Var = ['x', 'y']
#             P = [np.array([[1, 0, 0, 1] , [0, 1, 0, 0]])]
#             B = dnfTrue(2)
#             Q = [np.array([[1, -1, 1, 0]])]
#             T = genLItransitionrel(
#                 B, ([np.array([[1, 1, 1], [0, 1, 0], [0, 0, 1]])], dnfTrue(2)))
#             c = 2
#             d = 1

#         class b4:
#             Var = ['x', 'y']
#             P = [np.array([[1, 0, 0, 1] , [0, 1, 0, 0]])]
#             B = dnfTrue(2)
#             Q = [np.array([[1, -1, 1, 0]])]
#             T = genLItransitionrel(
#                 B, ([np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])], dnfTrue(2)))
#             c = 3
#             d = 1

#             '''  
#             For b4: this is invariant found:
#             Invariant Found:	((0*x + -1*y <= 0) /\ (-1*x + 1*y <= 0) /\ (-1*x + -1*y <= -1))
#                     Time Statistics:
#                         Total Time: 51.92798399899982
#                         Total Initialization and Re-initialization Time: 2.2624670229998856
#                         Total MCMC time: 48.84573149100004
#                         Total Z3 Time: 0.8197854849998976
#                         Total MCMC iterations forall threads: 423
#                         Total Z3 calls: 3
#                         Number of Threads: 4
#             '''
            

#         class b5:
#             Var = ['x', 'y']
#             P = [np.array([[1, 0, 0, 1] , [0, 1, 0, 0]])]
#             B = dnfTrue(2)
#             Q = [np.array([[1, -1, 1, 0]])]
#             T = genLItransitionrel(
#                 B, ([np.array([[3, 1, 0], [0, 1, 0], [0, 0, 1]])], dnfTrue(2)))
#             c = 2
#             d = 1

#         class b6:
#             Var = ['x', 'y']
#             P = [np.array([[1, 0, 0, 1] , [0, 1, 0, 0]])]
#             B = dnfTrue(2)
#             Q = [np.array([[1, -1, 1, 0]])]
#             T = genLItransitionrel(
#                 B, ([np.array([[1, 3, 0], [0, 1, 0], [0, 0, 1]])], dnfTrue(2)))
#             c = 2
#             d = 1

#         class b7:
#             Var = ['x', 'y']
#             P = [np.array([[1, 0, 0, 1] , [0, 1, 0, 0]])]
#             B = dnfTrue(2)
#             Q = [np.array([[1, -1, 1, 0]])]
#             T = genLItransitionrel(
#                 B, ([np.array([[4, 1, 0], [0, 1, 0], [0, 0, 1]])], dnfTrue(2)))
#             c = 3
#             d = 1
#         '''
#             For b7, invariant found:
#             Invariant Found:	((0*x + -1*y <= 0) /\ (-1*x + 1*y <= 0) /\ (0*x + 1*y <= 0))


#             Time Statistics:
#                 Total Time: 72.69041660200037
#                 Total Initialization and Re-initialization Time: 2.4581777250000414
#                 Total MCMC time: 69.45298488800017
#                 Total Z3 Time: 0.7792539890001535
#                 Total MCMC iterations forall threads: 560
#                 Total Z3 calls: 3
#                 Number of Threads: 4


#         '''

#         class b8:
#                 Var = ['x', 'y', 'z']
#                 P = [np.array([[1, 0, 0, 0, 1] , [0, 1, 0, 0, 0] , [0, 0, 1, 0, 0]])]
#                 B = dnfTrue(3)
#                 Q = [np.array([[1, -1, 0, 1, 0]])]
#                 T = genLItransitionrel(
#                     B, ([np.array([[3, 2, 3, 0], [0, 1, 1, 0], [0, 1, 2, 0], [0, 0, 0, 1]])], dnfTrue(3)))
#                 c = 3
#                 d = 1                
#         '''    
#             Invariant Found:        ((0*x + -2*y + 0*z <= 1) /\ (-1*x + 2*y + 2*z <= -1) /\ (0*x + 2*y + -2*z <= 1))


# Time Statistics:
#         Total Time: 171.6004012748599
#         Total Initialization and Re-initialization Time: 2.811329694930464
#         Total MCMC time: 166.8092993479222
#         Total Z3 Time: 1.9797722320072353
#         Total MCMC iterations forall threads: 3784
#         Total Z3 calls: 8
#         Number of Threads: 4
 
#             ''' 
#         class b9:
#             Var = ['x', 'y', 'z']
#             P = [np.array([[1, 0, 0, 0, 1] , [0, 1, 0, 0, 0] , [0, 0, 1, 0, 0]])]
#             B = dnfTrue(3)
#             Q = [np.array([[1, -1, 0, 1, 0]])]
#             T = genLItransitionrel(
#                 B, ([np.array([[1, 1, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])], dnfTrue(3)))
#             c = 3
#             d = 1        
#         '''
#         Invariant Found:        ((0*x + -1*y + 0*z <= 0) /\ (0*x + 0*y + -1*z <= 0) /\ (-1*x + 1*y + 0*z <= 0))




#         Time Statistics:
#                 Total Time: 111.60679428721778
#                 Total Initialization and Re-initialization Time: 2.7046414888463914
#                 Total MCMC time: 107.41555440216325
#                 Total Z3 Time: 1.4865983962081373
#                 Total MCMC iterations forall threads: 927
#                 Total Z3 calls: 5
#                 Number of Threads: 4


#         '''

#         class b10:
#             Var = ['u', 'x', 'y', 'z']
#             P = [np.array([[1, 0, 0, 0, 0, 3] , [0, 1, 0, 0, 0, 1] , [0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0]])]
#             B = dnfTrue(4)
#             Q = [np.array([[1, -1, -1, -3, 1, 0]])]
#             T = genLItransitionrel(
#                 B, ([np.array([[2654, 3, 9, 2, 2], [0, 1, 0, 0, 1], [0, 0, 1, 1, 0], [0, 0, 0, 2, 1] , [0, 0, 0, 0, 1]])], dnfTrue(4)))
#             c = 4
#             d = 1        

        ''' For other bx programs, where x != 4,7,8,9: Program finds invariant almost instantly.'''

        '''
            # Var = ( , )
                class standard_name:
                    Var = []
                    P = list3D_to_listof2Darrays([])
                    B = list3D_to_listof2Darrays([])
                    Q = dnfdisjunction(list3D_to_listof2Darrays([]) , B , 1)
                    T = genLItransitionrel(B, ( [ np.array([] ) ] , dnfTrue(len(Var))) )
            '''

        # class l1:
        #     Var = ['x', 'y']
        #     P = list3D_to_listof2Darrays([[[1, 0, 0, 0], [0, 1, 0, 1]]])
        #     B = list3D_to_listof2Darrays([[[1, 0, -1, 1000]]])
        #     Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 1, 1, 100]]]), B, 1)
        #     T = genLItransitionrel(B, ([np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]),], list3D_to_listof2Darrays([[[1, 0, -2, 100]]])),
        #                             ([np.array([[1, 0, -100], [0, 1, 1], [0, 0, 1]]),], list3D_to_listof2Darrays([[[1, 0, 1, 100]]])))
        #     c = 1
        #     d = 1



def input_to_repr(obj, c, d, c_list):
    if d is None:
        d = obj.d
    if c_list is None:
        if (not hasattr(obj, 'clist')):
            if (c is None):
                c = obj.c
            c_list = [c] * d
        else:
            c_list = obj.clist
    if (c is None):
        c = obj.c
    return Repr(obj.P, obj.B, obj.T, obj.Q, obj.Var, c, d, clist = c_list)


''''''''''''''''''''''''''''''''''''''
# Hill Climbing Algorithm:
# from hillclimbing import hill_climbing
# P = loop_lit.gj2007.P
# B = loop_lit.gj2007.B
# T = loop_lit.gj2007.T
# Q = loop_lit.gj2007.Q
# hill_climbing(Repr(P, B, T, Q))
