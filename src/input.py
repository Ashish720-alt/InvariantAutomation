# Testing
from dnfs_and_transitions import dnfTrue, list3D_to_listof2Darrays, dnfdisjunction, dnfnegation
import numpy as np
from repr import genLItransitionrel, Repr
from main import metropolisHastings


#IMP: Remember Q is Q \/ B for standard CHC
# LARGE INT is 1000000

class handcrafted:
    class mock:
        P = [np.array([[1, 0, 0]])]
        B = [np.array([[1, -1, 5]])]
        Q = [np.array([[1, -1, 6]])] 
        T = genLItransitionrel(B, ( [np.array([[1, 1], [0, 1]])] , dnfTrue(1) ) ) 

    # Variable vector: (x,y)
    class c2d1_1:
        P = [np.array([[1, 0, 0, 1], [0, 1, 0, 1] ])]
        B = dnfTrue(2)
        Q = [np.array([[1, -1, 1, 0], [0,1,1,1] ])]
        T = genLItransitionrel(B, ( [np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])] , dnfTrue(2) ) )         


class loop_lit:
    # Variable vector: (x,y) -- modified after 2 iterations
    class afnp2014_modified:
        P = [np.array([[1, 0, 0, 2], [0, 1, 0, 2] ])]
        B = [np.array([[0, 1, -2, 1000]])]
        Q = [np.array([[1, -1, 1, 0]])]
        T = genLItransitionrel(B, ( [np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])] , dnfTrue(2) ) ) 

    # Variable vector: (a,b,i,n)
    class bhmr2007:
        P = [np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, -1, 1000000] ])]
        B = [np.array([[0, 0, 1,-1, -1, 0]])]
        Q = [np.array([[1, 1, 0, -3, 0, 0]]), np.array([[0, 0, 1 ,-1, -1, 0]])]
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, 0, 1], [0, 1, 0, 0, 2], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]]) , 
                                       np.array([[1, 0, 0, 0, 2], [0, 1, 0, 0, 1], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]])  ] , dnfTrue(4) ) ) 
    
    # Variable vector: (i,j)
    class cggmp2005:
        P = [np.array([[1, 0, 0, 1], [0, 1, 0, 10] ])]
        B = [np.array([[-1, 1, 1, 0]])]
        Q = [np.array([[0, 1, 0, 6]]), np.array([[-1, 1, 1, 0]])]
        T = genLItransitionrel(B, ( [np.array([[1, 0, 2], [0, 1, -1], [0, 0, 1]])] , dnfTrue(2) ) )         

    # Variable vector: (lo, mid, hi)
    class cggmp2005_variant:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 0, 0], [0, 1, 0, 2, 0], [0, 1, 0, -2, 1000000], [0, -2, 1, 0, 0]]])
        B = list3D_to_listof2Darrays([[[0, 1, 0, 2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, 0, -1, 0, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1]] ) ] , dnfTrue(3)) )

    # Variable vector: (x,y)
    class gsv2008:
        P = [np.array([[1, 0, 0, -50], [0, 1, 2, 1000], [0,1,-2,1000000] ])]
        B = [np.array([[1, 0, -2, 0]])]
        Q = [np.array([[0, 1, 2, 0]]), np.array([[1, 0, -2, 0]])]
        T = genLItransitionrel(B, ( [np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])] , dnfTrue(2) ) )    

    # Variable vector: ['i', 'j', 'k']
    class css2003:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 0, 1], [0, 1, 0, 0, 1], [0, 0, 1, 1, 0], [0, 0, 1, -1, 1]]])
        B = list3D_to_listof2Darrays([[[1, 0, 0, -2, 1000000]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, 0, 1, -1, 2], [1, 0, 0, 1, 1], [-1, 0, -1, -1, -1]]]) , dnfnegation(B) , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 1, -1], [0, 0, 0, 1]] ) ] , dnfTrue(3)) )

    ##Conjunctive Randomized loop guard!
    #Variables:  ['x', 'y', 'z', 'w']
    class gcnr2008:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]]])
        B = list3D_to_listof2Darrays([[[0, 1, 0, 0, -2, 10000]]])
        Q = list3D_to_listof2Darrays([[[1, 0, 0, 0, 1, 4], [0, 1, 0, 0, -1, 2]]])
        T = genLItransitionrel(B, ([np.array( [[1, 0, 0, 0, 1], [0, 1, 0, 0, 100], [0, 0, 1, 0, 10], [0, 0, 0, 1, 1], [0, 0, 0, 0, 1]] ),
                                    np.array( [[1, 0, 0, 0, 1], [0, 1, 0, 0, 1], [0, 0, 1, 0, 10], [0, 0, 0, 1, 1], [0, 0, 0, 0, 1]] ) ] ,
                                    list3D_to_listof2Darrays([[[1, 0, 0, 0, 1, 4]]]) ) , 
                                  ([np.array( [[1, 0, 0, 0, 1], [0, 1, 0, 0, 100], [0, 0, 1, 0, 10], [0, 0, 0, 1, 1], [0, 0, 0, 0, 1]] ), 
                                    np.array( [[1, 0, 0, 0, 0], [0, -1, 0, 0, 0], [0, 0, 1, 0, 10], [0, 0, 0, 1, 1], [0, 0, 0, 0, 1]] )] , 
                                    list3D_to_listof2Darrays([[[-100, 0, 1, 0, 1, 0], [1, 0, 0, 0, -2, 4], [0, 1, 0, -10, 2, 0]]]) ) ,
                                  ([np.array([[1, 0, 0, 0, 1], [0, 1, 0, 0, 100], [0, 0, 1, 0, 10], [0, 0, 0, 1, 1], [0, 0, 0, 0, 1]] ),
                                    np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 10], [0, 0, 0, 1, 1], [0, 0, 0, 0, 1]] )] ,
                                    list3D_to_listof2Darrays([[[1, 0, 0, 0, -2, 4], [0, 1, 0, -10, -1, 0]], [[1, 0, 0, 0, -2, 4], [-100, 0, 1, 0, -2, 0]], 
                                                             [[1, 0, 0, 0, -2, 4], [1, 0, 0, 0, 1, 4]]]))
                              )

    # Variable vector: ['x', 'y']
    class gj2007:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 0], [0, 1, 0, 50]]])
        B = list3D_to_listof2Darrays([[[1, 0, -2, 100]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 1, 0, 100]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array( [[1, 0, 1], [0, 1, 0], [0, 0, 1]] )  ] , list3D_to_listof2Darrays([[[1, 0, -2, 50]]])  ),
                                  (  [ np.array( [[1, 0, 1], [0, 1, 1], [0, 0, 1]] ) ] ,  list3D_to_listof2Darrays([[[1, 0, 1, 50]]])  )
                              )                                           

    # Variable vector: ['x', 'y']
    class gj2007b:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 0], [0, 1, 0, 50]]])
        B = list3D_to_listof2Darrays([[[1, 0, -2, 100]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 1, 0, 100]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array( [[1, 0, 1], [0, 1, 0], [0, 0, 1]] )  ] , list3D_to_listof2Darrays([[[1, 0, -2, 50]]]) ), 
                                  ( [ np.array( [[1, 0, 1], [0, 1, 1], [0, 0, 1]] )  ] , list3D_to_listof2Darrays([[[1, 0, 1, 50]]]) )
                              )

    # Variable vector: ['x', 'y']
    class gr2006:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 0], [0, 1, 0, 0]]])
        B = list3D_to_listof2Darrays([[[1, 0, -2, 50]], [[0, 1, 1, 1]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, 0, 0, 100]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array( [[1, 0, 1], [0, 1, 1], [0, 0, 1]] ),  ] , list3D_to_listof2Darrays([[[1, 0, -2, 50], [0, 1, 1, 0]]]) ), 
                                  ( [ np.array( [[1, 0, 1], [0, 1, -1], [0, 0, 1]] ),  ] , list3D_to_listof2Darrays([[[1, 0, 1, 50], [0, 1, 1, 0]]]) )
                              )

    # Variables:  ['a', 'b', 'res', 'cnt']
    class hhk2008:
        P = list3D_to_listof2Darrays([[[-1, 0, 1, 0, 0, 0], [0, -1, 0, 1, 0, 0], [1, 0, 0, 0, -1, 1000000], [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, -1, 1000000]]])
        B = list3D_to_listof2Darrays([[[0, 0, 0, 1, 2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[-1, -1, 1, 0, 0, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, -1], [0, 0, 0, 0, 1]] ) ] , dnfTrue(4)) )    

    # Why does this give a deprecated numpy warning?
    # Variable vector: ['i', 'j', 'x', 'y']
    class jm2006:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0], [-1, 0, 1, 0, 0, 0], [0, -1, 0, 1, 0, 0]]])
        B = list3D_to_listof2Darrays([[[0, 0, 1, 0, 2, 0]], [[0, 0, 1, 0, -2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays( [[[1, -1, 0, 0, -1, 0], [1, -1, 0, 0, 1, 0]], [[0, 0, 0, 1, 0, 0]] ] ) , B , 1)
        T = genLItransitionrel(B, ( [ np.array( [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, -1], [0, 0, 0, 1, -1], [0, 0, 0, 0, 1]] ) ] , dnfTrue(4)) )

    # Variable vector: ['i', 'j', 'x', 'y', 'z']
    class jm2006_variant:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, -1, 1000000], [0, 1, 0, 0, 0, 1, 0], [-1, 0, 1, 0, 0, 0, 0], [0, -1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]]])
        B = list3D_to_listof2Darrays([[[0, 0, 1, 0, 0, 2, 0]], [[0, 0, 1, 0, 0, -2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, -1, 0, 0, 0, -1, 0], [1, -1, 0, 0, 0, 1, 0]], [[0, 0, 0, 1, 1, 0, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, -1], [0, 0, 0, 1, 0, -1], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 1]] ) ] , dnfTrue(5)) )


class loop_new:

    # Variable vector: ['i']
    class count_by_1:
        P = list3D_to_listof2Darrays([[[1, 0, 0]]])
        B = list3D_to_listof2Darrays([[[1, -2, 1000000]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, 0, 1000000]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 1], [0, 1]] ) ] , dnfTrue(1)) )    

    # Variable vector: ['i']
    class count_by_1_variant:
        P = list3D_to_listof2Darrays([[[1, 0, 0]]])
        B = list3D_to_listof2Darrays([[[1, -2, 1000000]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, -1, 1000000]]]) , dnfnegation(B) , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 1], [0, 1]] ) ] , dnfTrue(1)) )    

    # Variable vector: ['i']
    class count_by_2:
        P = list3D_to_listof2Darrays([[[1, 0, 0]]])
        B = list3D_to_listof2Darrays([[[1, -2, 1000000]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, 0, 1000000]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 2], [0, 1]] ) ] , dnfTrue(1)) )    

    # It never finds the rotation neighbors here, because one coefficient is 10,00000
    # Variable vector: ['i', 'k']
    class count_by_k:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, -1, 10]]])
        B = list3D_to_listof2Darrays([[[1, -1000000, -2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, -1000000, 0, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]] ) ] , dnfTrue(2)) )


class loop_simple:
    class nested_1:
        P = [np.array([[1, 0, 0]])]
        B = [np.array([[1, -2, 6]])]
        Q = [np.array([[1, -1, 6]])] 
        T = genLItransitionrel(B, ( [np.array([[1, 1], [0, 1]])] , dnfTrue(1) ) ) 

class loop_zilu:
    # Variable vector: (x , y)
    class benchmark01_conjunctive:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 1], [0, 1, 0, 1]]])
        B = list3D_to_listof2Darrays([])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 1, 1, 1]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array( [[1, 1, 0], [1, 1, 0], [0, 0, 1]] ) ] , dnfTrue(2)) )

    # Variables:  ['n', 'i', 'l']
    class benchmark02_linear:
        P = list3D_to_listof2Darrays([[[0, 0, 1, 2, 0], [0, 1, -1, 0, 0]]])
        B = list3D_to_listof2Darrays([[[-1, 1, 0, -2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 0, 1, 1, 1]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]] ) ] , dnfTrue(3)) )        

    # Variables:  ['x', 'y', 'i', 'j', 'flag']
    class benchmark03_linear:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]]])
        B = list3D_to_listof2Darrays([[[0, 0, 0, 0, 1, -2, 0]], [[0, 0, 0, 0, 1, 2, 0]]])
        Q = list3D_to_listof2Darrays([[[0, 0, -1, 1, 0, 1, 0]]]) 
        T = genLItransitionrel(B, 
                                ( [ np.array([[1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 1], 
                                        [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[0, 0, 0, 0, 1, 0, 0]]])) , 
                                ( [ np.array([[1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 2], 
                                        [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[0, 0, 0, 0, 1, -2, 0]], [[0, 0, 0, 0, 1, 2, 0]]]))         
                                ) 

    # Variables:  ['k', 'j', 'n']
    class benchmark04_conjunctive:
        P = list3D_to_listof2Darrays([[[0, 0, 1, 1, 1], [1, 0, -1, 1, 0], [0, 1, 0, 0, 0]]])
        B = list3D_to_listof2Darrays([[[0, 1, -1, -1, -1]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, 0, 0, 1, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, -1], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]] ) ] , dnfTrue(3)) )

    # Variables:  ['x', 'y', 'n']
    class benchmark05_conjunctive:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 1, 0], [1, -1, 0, -1, 0], [0, 1, -1, -2, 0]]])
        B = list3D_to_listof2Darrays([[[1, 0, -1, -2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 1, -1, 0, 0]]]) , B , 1)
        T = genLItransitionrel(B,
                               ( [ np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) ] 
                                   , list3D_to_listof2Darrays([[[1, -1, 0, -1, 0]]]) ), 
                               ( [ np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]) ] 
                                   , list3D_to_listof2Darrays([[[1, -1, 0, 2, 0]]]) ),                                    
                                    )

    # Variables:  ['i', 'j', 'x', 'y', 'k']
    class benchmark06_conjunctive:
        P = list3D_to_listof2Darrays([[[0, 0, 1, 1, -1, 0, 0], [0, 1, 0, 0, 0, 0, 0]]])
        B = dnfTrue(5)
        Q = list3D_to_listof2Darrays([[[0, 0, 1, 1, -1, 0, 0]]]) 
        T = genLItransitionrel(B, 
                                ( [ np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, -1], [0, 0, 0, 0, 1, 0], 
                                  [0, 0, 0, 0, 0, 1]]) ] , list3D_to_listof2Darrays([[[-1, 1, 0, 0, 0, 0, 0]]]) ),
                                ( [ np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, -1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 0], 
                                  [0, 0, 0, 0, 0, 1]]) ] , list3D_to_listof2Darrays([[[-1, 1, 0, 0, 0, -2, 0]], [[1, -1, 0, 0, 0, -2, 0]]]) )   )

    # Variables:  ['i', 'n', 'k', 'flag']
    class benchmark07_linear:
        P = list3D_to_listof2Darrays([[[0, 1, 0, 0, 2, 0], [0, 1, 0, 0, -2, 10], [0, 0, 0, 1, 0, 0]], [[0, 1, 0, 0, 2, 0], [0, 1, 0, 0, -2, 10], [0, 0, 0, 1, 0, 1]]])
        B = list3D_to_listof2Darrays([[[1, -1, 0, 0, -2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, -1, 1, 0, 2, 0]]]) , B , 1)
        T = genLItransitionrel(B, 
                            ( [ np.array( [[1, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 1, 0, 4000], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]] ) ] , 
                                    list3D_to_listof2Darrays([[[0, 0, 0, 1, 0, 1]]])  ), 
                            (  [ np.array([[1, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 1, 0, 2000], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]  ) ] ,      
                                    list3D_to_listof2Darrays([[[0, 0, 0, 1, 0, 0]]]) )   )     

    # Variables:  ['n', 'sum', 'i']
    class benchmark08_conjunctive:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]])
        B = list3D_to_listof2Darrays([[[-1, 0, 1, -2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 1, 0, 1, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]] ) ] , dnfTrue(3)) )

    # Variables:  ['x', 'y']
    class benchmark09_conjunctive:
        P = list3D_to_listof2Darrays([[[1, -1, 0, 0], [0, 1, 1, 0]]])
        B = list3D_to_listof2Darrays([[[0, 1, -2, 0]], [[0, 1, 2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 1, 0, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, -1], [0, 1, -1], [0, 0, 1]]) ] , dnfTrue(2)) )

    # Variable vector: ( 'i , c)
    class benchmark10_conjunctive:
        P = list3D_to_listof2Darrays([[[0, 1, 0, 0], [1, 0, 0, 0]]])
        B = list3D_to_listof2Darrays([[[1, 0, -2, 100], [1, 0, 2, -1]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 1, 1, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1]] ) ] , dnfTrue(2)) )

    # Variables:  ['x', 'n']
    class benchmark11_linear:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 0], [0, 1, 2, 0]]])
        B = list3D_to_listof2Darrays([[[1, -1, -2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, -1, 0, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]] ) ] , dnfTrue(2)) )

    # Variables:  ['x', 'y', 't']
    class benchmark12_linear:
        P = list3D_to_listof2Darrays([[[1, -1, 0, 2, 0], [0, 1, -1, 0, 0]], [[1, -1, 0, -2, 0], [0, 1, -1, 0, 0]]])
        B = dnfTrue(3)
        Q = list3D_to_listof2Darrays([[[0, 1, -1, 1, 0]]]) 
        T = genLItransitionrel(B, 
                        ( [np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, 0, 2, 0]]]) ) ,
                        ( [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, 0, -1, 0]]]) ) 
                        )

    # Variables:  ['i', 'j', 'k']
    class benchmark13_conjunctive:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]])
        B = list3D_to_listof2Darrays([[[1, 0, -1, -1, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[-1, 1, 0, 0, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]] ) ] , dnfTrue(3)) )

    # Variables:  ['i']
    class benchmark14_linear:
        P = list3D_to_listof2Darrays([[[1, 1, 0], [1, -1, 200]]])
        B = list3D_to_listof2Darrays([[[1, 2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, 1, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, -1], [0, 1]] ) ] , dnfTrue(1)) )

    # Variables:  ['low', 'mid', 'high']
    class benchmark15_conjunctive:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 0, 0], [0, 1, 0, 1, 1], [0, -2, 1, 0, 0]]])
        B = list3D_to_listof2Darrays([[[0, 1, 0, 2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, 0, -1, 0, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1]] ) ] , dnfTrue(3)) )

    # Variables:  ['i', 'k']
    class benchmark16_conjunctive:
        P = list3D_to_listof2Darrays([[[0, -1, -1, 0], [0, 1, -1, 1], [1, 0, 0, 1]]])
        B = dnfTrue(2)
        Q = list3D_to_listof2Darrays([[[-1, -1, -1, -1], [1, 1, -1, 2], [1, 0, 1, 1]]]) 
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 1], [0, 1, -1], [0, 0, 1]] ) ] , dnfTrue(2)) )

    # Variables:  ['i', 'k', 'n']
    class benchmark17_conjunctive:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]])
        B = list3D_to_listof2Darrays([[[1, 0, -1, -2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 1, -1, 1, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]] ) ] , dnfTrue(3)) )

    # Variables:  ['i', 'k', 'n']
    class benchmark18_conjunctive:
        P = list3D_to_listof2Darrays([[[1, 0, -1, 0, 0], [0, 1, -1, 0, 0], [0, 0, 1, 2, 0]]])
        B = list3D_to_listof2Darrays([[[0, 0, 1, 2, 0], [1, 0, 0, 2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 1, 0, 0, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, 0], [0, 0, 0, 1]] ) ] , dnfTrue(3)) )

    # Variables:  ['j', 'k', 'n']
    class benchmark19_conjunctive:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 2, 0]]])
        B = list3D_to_listof2Darrays([[[1, 0, -1, -2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 1, -1, 0, 0], [1, -1, 0, 0, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]] ) ] , dnfTrue(3)) )

    # Variables:  ['i', 'n', 'sum']
    class benchmark20_conjunctive:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 1, 0, -1, 100], [0, 0, 1, 0, 0]]])
        B = list3D_to_listof2Darrays([[[1, -1, 0, -2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 0, 1, 1, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]] ) ] , dnfTrue(3)) )

    # Variables:  ['x', 'y']
    class benchmark21_disjunctive:
        P = list3D_to_listof2Darrays([[[0, 1, 2, 0]], [[1, 0, 2, 0]]])
        B = list3D_to_listof2Darrays([[[1, 1, -1, -2]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 1, 2, 0]], [[1, 0, 2, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]] ) ]  , list3D_to_listof2Darrays( [[[1, 0, 2, 0]]] ) )  ,
                                 ( [ np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]] ) ] , list3D_to_listof2Darrays( [[[1, 0, -1, 0]]]) ) ,
                              )

    # Variables:  ['x', 'y']
    class benchmark22_conjunctive:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 1], [0, 1, 0, 0]]])
        B = dnfTrue(2)
        Q = list3D_to_listof2Darrays([[[1, 0, 0, 1], [0, 1, 0, 0]]]) 
        T = genLItransitionrel(B, ( [ np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]] ) ] , dnfTrue(2)) )

    # Variables:  ['i', 'j']
    class benchmark23_conjunctive:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 0], [0, 1, 0, 0]]])
        B = list3D_to_listof2Darrays([[[1, 0, -2, 100]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 1, 0, 200]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 2], [0, 1, 2], [0, 0, 1]] ) ] , dnfTrue(2)) )

    # Variables:  ['i', 'k', 'n']
    class benchmark24_conjunctive:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 0, 0], [0, 1, -1, 0, 0], [0, 0, 1, 1, 0]]])
        B = list3D_to_listof2Darrays([[[1, 0, -1, -2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 2, -1, 1, -1]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, 2], [0, 1, 0, -1], [0, 0, 1, 0], [0, 0, 0, 1]] ) ] , dnfTrue(3)) )

    # Variables:  ['x']
    class benchmark25_linear:
        P = list3D_to_listof2Darrays([[[1, -2, 0]]])
        B = list3D_to_listof2Darrays([[[1, -2, 10]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, 0, 10]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 1], [0, 1]] ) ] , dnfTrue(1)) )

    # Variables:  ['x', 'y']
    class benchmark26_linear:
        P = list3D_to_listof2Darrays([[[1, -1, -2, 0]]])
        B = list3D_to_listof2Darrays([[[1, -1, -2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, -1, 0, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]] ) ] , dnfTrue(2)) )

    # Variables:  ['i', 'j', 'k']
    class benchmark27_linear:
        P = list3D_to_listof2Darrays([[[1, -1, 0, -2, 0], [-1, 1, 1, 2, 0]]])
        B = list3D_to_listof2Darrays([[[1, -1, 0, -2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 0, 1, 2, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]] ) ] , dnfTrue(3)) )

    # Variables:  ['i', 'j']
    class benchmark28_linear:
        P = list3D_to_listof2Darrays([[[1, -1, 2, 0], [1, 1, -2, 0]]])
        B = list3D_to_listof2Darrays([[[1, -1, -2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[-1, 1, 0, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[-1, 1, 0], [1, 0, 0], [0, 0, 1]] ) ] , list3D_to_listof2Darrays( [[[-2, 1, -2, 0]]] ) ) , 
                                ( [ np.array([[1, 0, 0], [-1, 1, 0], [0, 0, 1]] ) ] , list3D_to_listof2Darrays( [[[-2, 1, 1, 0]]] )    )     )

    # Variables:  ['x', 'y']
    class benchmark29_linear:
        P = list3D_to_listof2Darrays([[[1, -1, -2, 0]]])
        B = list3D_to_listof2Darrays([[[1, -1, -2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, -1, 1, 0], [1, -1, -1, 99]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 100], [0, 1, 0], [0, 0, 1]] ) ] , dnfTrue(2)) )

    # Variables:  ['x', 'y']
    class benchmark30_conjunctive:
        P = list3D_to_listof2Darrays([[[-1, 1, 0, 0]]])
        B = list3D_to_listof2Darrays([])
        Q = list3D_to_listof2Darrays([[[1, -1, 0, 0]]]) 
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]] ) ] , dnfTrue(2)) )

    # Variables:  ['x', 'y']
    class benchmark31_disjunctive:
        P = list3D_to_listof2Darrays([[[1, 0, -2, 0]]])
        B = list3D_to_listof2Darrays([[[1, 0, -2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 1, 1, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]] ) ] , dnfTrue(2)) )

    # Variables:  ['x']
    class benchmark32_linear:
        P = list3D_to_listof2Darrays(  [[[1, 0, 1]], [[1, 0, 2]]])
        B = dnfTrue(1)
        Q = list3D_to_listof2Darrays([[[1, -1, 8]]]) 
        T = genLItransitionrel(B, ( [ np.array([[0, 2], [0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, 1]]]) ) ,
                                  ( [ np.array([[0, 1], [0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, 2]]])) ,
                                  ( [ np.array([[1, 0], [0, 1]] ) ] , list3D_to_listof2Darrays( [ [[1, -1, 0]] , [[1, 1, 3]] ]))
            )

    # Variables:  ['x']
    class benchmark33_linear:
        P = list3D_to_listof2Darrays([[[1, 1, 0]]])
        B = list3D_to_listof2Darrays([[[1, -2, 100], [1, 1, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, 1, 100]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 1], [0, 1]] ) ] , dnfTrue(1)) )

    #Variables:  ['j', 'k', 'n']
    class benchmark34_conjunctive:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 0, 0], [0, 1, -1, 0, 0], [0, 0, 1, 2, 0]]])
        B = list3D_to_listof2Darrays([[[0, 0, 1, 2, 0], [1, 0, -1, -2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 1, 0, 0, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 0], [0, 0, 0, 1]] ) ] , dnfTrue(3)) )

    # Variables:  ['x']
    class benchmark35_linear:
        P = list3D_to_listof2Darrays( [[[1, 1, 0]]])
        B = list3D_to_listof2Darrays([[[1, 1, 0], [1, -2, 10]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, 1, 10]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 1], [0, 1]] ) ] , dnfTrue(1)) )

    # Variables:  ['x', 'y']
    class benchmark36_conjunctive:
        P = list3D_to_listof2Darrays( [[[1, -1, 0, 0], [0, 1, 0, 0]]])
        B = dnfTrue(2)
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, -1, 0, 0], [1, 0, 1, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]] ) ] , dnfTrue(2)) )

    # Variables:  ['x', 'y']
    class benchmark37_conjunctive:
        P = list3D_to_listof2Darrays([[[1, -1, 0, 0], [1, 0, 1, 0]]])
        B = list3D_to_listof2Darrays([[[1, 0, 2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 1, 1, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, -1], [0, 1, -1], [0, 0, 1]] ) ] , dnfTrue(2)) )

    # Variables:  ['x', 'y']
    class benchmark38_conjunctive:
        P = list3D_to_listof2Darrays([[[1, -1, 0, 0], [0, 1, 0, 0]]])
        B = dnfTrue(2)
        Q = list3D_to_listof2Darrays([[[1, -4, 0, 0], [1, 0, 1, 0]]]) 
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 4], [0, 1, 1], [0, 0, 1]] ) ] , dnfTrue(2)) )

    # Variables:  ['x', 'y']
    class benchmark39_conjunctive:
        P = list3D_to_listof2Darrays([[[1, -4, 0, 0], [1, 0, 1, 0]]])
        B = list3D_to_listof2Darrays([[[1, 0, 2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 1, 1, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, -4], [0, 1, -1], [0, 0, 1]] ) ] , dnfTrue(2)) )

    # Variables:  ['x', 'y']
    class benchmark40_polynomial:
        P = list3D_to_listof2Darrays([[[1, 0, -1, 0], [0, 1, -1, 0]]])
        B = dnfTrue(2)
        Q = list3D_to_listof2Darrays([[[1, 0, -1, 0], [0, 1, -1, 0]]]) 
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, 2, 0]]]) ) ,
                                  ( [ np.array([[1, 0, -1], [0, 1, 0], [0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, -2, 0]]]) ) ,
                                  ( [ np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, 0, 0], [0, 1, 2, 0]]]) ) ,
                                  ( [ np.array([[1, 0, -1], [0, 1, 0], [0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, 0, 0], [0, 1, -1, 0]]]) ) 
        )

    # Variables:  ['x', 'y', 'z']
    class benchmark41_conjunctive:
        P = list3D_to_listof2Darrays([[[1, -1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]])
        B = dnfTrue(3)
        Q = list3D_to_listof2Darrays([[[1, -1, 0, 0, 0], [1, 0, 0, 1, 0], [1, 1, 1, 0, 0]]]) 
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, -2], [0, 0, 0, 1]] ) ] , dnfTrue(3)) )

    # Variables:  ['x', 'y', 'z']
    class benchmark42_conjunctive:
        P = list3D_to_listof2Darrays([[[1, -1, 0, 0, 0], [1, 0, 0, 1, 0], [1, 1, 1, 0, 0]]])
        B = list3D_to_listof2Darrays([[[1, 0, 0, 2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 0, 1, -1, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, 2], [0, 0, 0, 1]] ) ] , dnfTrue(3)) )

    # Variables:  ['x', 'y']
    class benchmark43_conjunctive:
        P = list3D_to_listof2Darrays([[[1, 0, -2, 100], [0, 1, -2, 100]]])
        B = list3D_to_listof2Darrays([[[1, 0, -2, 100], [0, 1, -2, 100]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, 0, 0, 100]], [[0, 1, 0, 100]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]] ) ] , dnfTrue(2)) )

    # Variables:  ['x', 'y']
    class benchmark44_disjunctive:
        P = list3D_to_listof2Darrays([[[1, 0, -2, 100], [0, 1, -2, 100]]])
        B = list3D_to_listof2Darrays([[[1, 0, -2, 100], [0, 1, -2, 100]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, 0, 0, 100]], [[0, 1, 0, 100]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]] ) ] , dnfTrue(2)) )

    # Variables:  ['x', 'y']
    class benchmark45_disjunctive:
        P = list3D_to_listof2Darrays([[[0, 1, 2, 0]], [[1, 0, 2, 0]]])
        B = dnfTrue(2)
        Q = list3D_to_listof2Darrays([[[1, 0, 2, 0]], [[0, 1, 2, 0]]]) 
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, 2, 0]]]) ) , 
                                  ( [ np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, -1, 0]]]) )
                                 )

    # Variables:  ['x', 'y', 'z']
    class benchmark46_disjunctive:
        P = list3D_to_listof2Darrays([[[0, 1, 0, 2, 0]], [[1, 0, 0, 2, 0]], [[0, 0, 1, 2, 0]]])
        B = dnfTrue(3)
        Q = list3D_to_listof2Darrays([[[0, 1, 0, 2, 0]], [[1, 0, 0, 2, 0]], [[0, 0, 1, 2, 0]]]) 
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, 0, 2, 0]]]) ) , 
                                  ( [ np.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[0, 1, 0, 2, 0], [1, 0, 0, -1, 0]]]) ) , 
                                  ( [ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[0, 1, 0, -1, 0], [1, 0, 0, -1, 0]]]) ) 
                                   )

    # Variables:  ['x', 'y']
    class benchmark47_linear:
        P = list3D_to_listof2Darrays([[[1, -1, -2, 0]]])
        B = list3D_to_listof2Darrays([[[1, -1, -2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, -1, 1, 0], [1, -1, -1, 16]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 7], [0, 1, -10], [0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, -2, 0], [0, 1, -2, 0]]]) ),
                                  ( [ np.array([[1, 0, 10], [0, 1, -10], [0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, 1, 0], [0, 1, -2, 0]]]) ), 
                                  ( [ np.array([[1, 0, 7], [0, 1, 3], [0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, -2, 0], [0, 1, 1, 0]]]) ), 
                                  ( [ np.array([[1, 0, 10], [0, 1, 3], [0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, 1, 0], [0, 1, 1, 0]]]) )
                                  )

    # Variables:  ['i', 'j', 'k']
    class benchmark48_linear:
        P = list3D_to_listof2Darrays([[[1, -1, 0, -2, 0], [0, 0, 1, 2, 0]]])
        B = list3D_to_listof2Darrays([[[1, -1, 0, -2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, -1, 1, 2, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]] ) ] , dnfTrue(3)) )

    # Variables:  ['i', 'j', 'r']
    class benchmark49_linear:
        P = list3D_to_listof2Darrays([[[-1, -1, 1, 2, 0]]])
        B = list3D_to_listof2Darrays([[[1, 0, 0, 2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[-1, -1, 1, 2, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0, -1], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]] ) ] , dnfTrue(3)) )

    # Variables:  ['xa', 'ya']
    class benchmark50_linear:
        P = list3D_to_listof2Darrays([[[1, 1, 2, 0]]])
        B = list3D_to_listof2Darrays([[[1, 0, 2, 0]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 1, 1, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, -1], [0, 1, 1], [0, 0, 1]] ) ] , dnfTrue(2)) )

    # Variables:  ['x']
    class benchmark51_polynomial:
        P = list3D_to_listof2Darrays([[[1, 1, 0], [1, -1, 50]]])
        B = dnfTrue(1)
        Q = list3D_to_listof2Darrays([[[1, 1, 0], [1, -1, 50]]]) 
        T = genLItransitionrel(B, ( [ np.array([[1, 1], [0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, 0]], [[1, 2, 50]]]) ) , 
                                   ( [ np.array([[1, -1], [0, 1]] ) ] , list3D_to_listof2Darrays([[[1, -2, 0]], [[1, 2, 0], [1, -1, 50]]]) ) 
                                )

    # Variables:  ['i']
    class benchmark52_polynomial:
        P = list3D_to_listof2Darrays([[[1, -2, 10], [1, 2, -10]]])
        B = list3D_to_listof2Darrays([[[1, -2, 10], [1, 2, -10]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, 0, 10]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 1], [0, 1]] ) ] , dnfTrue(1)) )

    # Variables:  ['x', 'y']
    class benchmark53_polynomial:
        P = list3D_to_listof2Darrays( [[[1, 0, -1, 0], [0, 1, -1, 0]]])
        B = dnfTrue(2)
        Q = list3D_to_listof2Darrays([[[1, 0, -1, 0], [0, 1, -1, 0]]]) 
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, 2, 0]]])) , 
                                  ( [ np.array([[1, 0, -1], [0, 1, 0], [0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, -2, 0]]])  ) ,
                                  ( [ np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, 0, 0], [0, 1, 2, 0]]])  ) ,
                                  ( [ np.array([[1, 0, -1], [0, 1, 0], [0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, 0, 0], [0, 1, -1, 0]]])  ) 
         )


class loops_crafted_1:

    # Variables:  ['x']
    class Mono1_1_1:
        P = list3D_to_listof2Darrays([[[1, 0, 0]]])
        B = list3D_to_listof2Darrays([[[1, -2, 100000000]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, 0, 100000001]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 1], [0, 1]] ) ] , list3D_to_listof2Darrays([[[1, -2, 10000000]]]) ) ,
                                  ( [ np.array([[1, 2], [0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 1, 10000000]]]) ) )

    # Variables:  ['x']
    class Mono1_1_2:
        P = list3D_to_listof2Darrays([[[1, 0, 0]]])
        B = list3D_to_listof2Darrays([[[1, -2, 100000000]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, 0, 100000000]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 1], [0, 1]] ) ] , list3D_to_listof2Darrays([[[1, -2, 10000000]]]) ) ,
                                  ( [ np.array([[1, 2], [0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 1, 10000000]]]) ) )

    # Variables:  ['x', 'y']
    class Mono3_1:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 0], [0, 1, 0, 0]]])
        B = list3D_to_listof2Darrays([[[1, 0, -2, 1000000]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 1, 2, 0]], [[0, 1, -2, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, -2, 500000]]]) ), 
                                  ( [ np.array([[1, 0, 1], [0, 1, -1], [0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, 1, 500000]]]) )  )

    # Variables:  ['x', 'y']
    class Mono4_1:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 0], [0, 1, 0, 0]]])
        B = list3D_to_listof2Darrays([[[1, 0, -2, 1000000]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[0, 1, 2, 0]], [[0, 1, -2, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, -2, 500000]]]) ), 
                                  ( [ np.array([[1, 0, 1], [0, 1, -1], [0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, 1, 500000]]]) )  )

    # Variables:  ['x', 'y']
    class Mono4_1:
        P = list3D_to_listof2Darrays([[[1, 0, 0, 0], [0, 1, 0, 500000]]])
        B = list3D_to_listof2Darrays([[[1, 0, -2, 1000000]]])
        Q = dnfdisjunction(list3D_to_listof2Darrays([[[1, -1, -2, 0]], [[-1, 1, -2, 0]]]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, -2, 500000]]]) ), 
                                  ( [ np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]] ) ] , list3D_to_listof2Darrays([[[1, 0, 1, 500000]]]) ) )

    '''
    # Variable vector: ( , )
    class standard_name:
        P = list3D_to_listof2Darrays([])
        B = list3D_to_listof2Darrays([])
        Q = dnfdisjunction(list3D_to_listof2Darrays([]) , B , 1)
        T = genLItransitionrel(B, ( [ np.array([] ) ] , dnfTrue(n)) )
    '''
    


def classwrapper_metropolisHastings (obj):
    metropolisHastings(Repr(obj.P, obj.B, obj.T, obj.Q))


classwrapper_metropolisHastings(loops_crafted_1.Mono4_1)

''''''''''''''''''''''''''''''''''''''
# Hill Climbing Algorithm:
# from hillclimbing import hill_climbing
# P = loop_lit.gj2007.P
# B = loop_lit.gj2007.B
# T = loop_lit.gj2007.T
# Q = loop_lit.gj2007.Q
# hill_climbing(Repr(P, B, T, Q))