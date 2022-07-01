import repr
from input import get_input

''' Clause Systems:
SSL, DL = (defined in loop_simple.py)


SL, Variant 1:
P -> I
I(s) /\ B /\ T(s,t) -> I(t)
I /\ B -> M

'''


class loop_new:

    #SSL, 1 parameter LARGE_INT=1000000
    count_by_1 = get_input(P=np.array([[[1, 0, 0]]]),
                B=np.array([[[1, -2, 1000000]]]),
                Q=np.array([[[1, 0, 1000000]]]),
                T=repr.SimpleTotalTransitionFunc(np.array([[1, 1], [0, 1]]))) 

    #SL, Variant 1, 1 parameter LARGE_INT=1000000
    count_by_1_variant = get_input(P=np.array([[[1, 0, 0]]]),
                B=np.array([[[1, -2, 1000000]], [[1, 2, 1000000]] ]),
                M=np.array([[[1, -1, 1000000]]]),
                T=repr.SimpleTotalTransitionFunc(np.array([[1, 1], [0, 1]])))     

    #SSL, 1 parameter LARGE_INT=1000000
    count_by_2 = get_input(P=np.array([[[1, 0, 0]]]),
                B=np.array([[[1, -2, 1000000]]]),
                Q=np.array([[[1, 0, 1000000]]]),
                T=repr.SimpleTotalTransitionFunc(np.array([[1, 2], [0, 1]])))

    #SSL, 1 parameter LARGE_INT=1000000, variable_vector = (i, k)
    count_by_k = get_input(P=np.array([[[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, -1, 10] ]]),
                B=np.array([[[1, -1000000, -2, 0]]]),
                Q=np.array([[[1, -1000000, 0, 0]]]),
                T=repr.SimpleTotalTransitionFunc(np.array([[1, 1, 0], [0, 1, 0],  [0, 0, 1]]))) 

    count_by_nondet.c = ''' Random Initialization T , How can I write this?'''

    #Actual guass_sum has quadratic Q! SSL, variable_vector = (i, n, sum)
    gauss_sum_adapted = get_input(P=np.array([[[1, 0, 0, 0, 1], [0, 0, 1, 0, 0], [0, 1, 0, 1, 1], [0, 1, 0, -1, 1000]]]),
                    B=np.array([[[1, -1, 0, -2, 0]]]),
                    Q=np.array([[[0, -1, 1, 1, 0]]]),
                    T=repr.SimpleTotalTransitionFunc(np.array([[1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]])))

    #Actual program has quadratic Q! SSL, originally goto variation, but same as gauss_sum_adapted
    gauss_sum.i.p+cfa-reducer_adapted = get_input(P=np.array([[[1, 0, 0, 0, 1], [0, 0, 1, 0, 0], [0, 1, 0, 1, 4], [0, 1, 0, -1, 1000]]]),
                    B=np.array([[[1, -1, 0, -2, 0]]]),
                    Q=np.array([[[0, -1, 1, 1, 0]]]),
                    T=repr.SimpleTotalTransitionFunc(np.array([[1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]])))

    #Actual program has quadratic Q! SSL, originally goto variation, but slightly different than gauss_sum_adapted
    gauss_sum.i.p+lhb-reducer_adapted = get_input(P=np.array([[[1, 0, 0, 0, 5], [0, 0, 1, 0, 10], [0, 1, 0, 1, 1], [0, 1, 0, -1, 1000]]]),
                    B=np.array([[[1, -1, 0, -2, 0]]]),
                    Q=np.array([[[0, -1, 1, 1, 0]]]),
                    T=repr.SimpleTotalTransitionFunc(np.array([[1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]])))

    #SSL
    half = ''' Need operator modulus %, to represent this ''' #Also allow for operator '!=' ?

    #DL, variable_vector = (i,j,k,n,m)
    nested-1 = get_input(P=np.array([[[1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 10], [0, 0, 0, 1, 0, -1, 1000] , [0, 0, 0, 0, 1, 1, 10], [0, 0, 0, 0, 1, -1, 1000]]]),
                    B1=np.array([[[1, 0, 0, 1, 0, -2, 0]]]),
                    B2=np.array([[[0, 1, 0, 0, 1, -2, 0]]]),
                    Q=np.array([[[0, 0, 1, 0, 0, 1, 100]]]),
                    T1=repr.SimpleTotalTransitionFunc(np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0] , [0, 0, 1, 0, 0, 0] , [0, 0, 0, 1, 0, 0] ,[0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1]])),
                    T2=repr.SimpleTotalTransitionFunc(np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0] , [0, 0, 1, 0, 0, 1] , [0, 0, 0, 1, 0, 0] ,[0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1]])),
                    T3=repr.SimpleTotalTransitionFunc(np.array([[1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0] , [0, 0, 1, 0, 0, 0] , [0, 0, 0, 1, 0, 0] ,[0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1]])) )
