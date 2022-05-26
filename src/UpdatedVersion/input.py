""" Read programs as input and extract the logic formulas.
The module is left as empty right now as we rely on manual input for now.
"""

import numpy as np
import repr


''' IMP: To calculate T, we need to resolve RAW and WAW dependencies. Eg
i = i + 1
sum = sum + i


Then : i_new = i_old + 1
And : sum_new = sum_old + i_old + 1 (as this is i_new)
'''


# def get_input(P, B, Q, T):
#     return repr.Repr(P, B, Q, T)

def get_input(CHC, DNFs, trans_funcs, invs):
    return repr.Repr(CHC, DNFs, trans_funcs, invs)

# Still need to do symbolic execution on your own
# def get_transition_matrix( T, var ):
#     n = len(var)
#     for t in T:


class mock:
    # mock1 = get_input(P=np.array([[[1, 0, 0, 0, 1], [0, 0, 1, 0, 0], [0, 1, 0, 1, 1], [0, 1, 0, -1, 1000]]]),
    #                 B=np.array([[[1, -1, 0, -2, 0]]]),
    #                 Q=np.array([[[0, -1, 1, 1, 0]]]),
    #                 T=repr.SimpleTotalTransitionFunc(repr.I(3) + repr.E(3,(1,4)) + repr.E(3,(3,1)) )  )


    # mock2 = get_input(P=np.array([[[1, 0, 0]]]),
    #                 B=np.array([[[1, -2, 6]]]),
    #                 Q=np.array([[[1, 0, 6]]]),
    #                 T=repr.SimpleTotalTransitionFunc(repr.I(1) + repr.E(1,(1,2)) ))

    # mock3 = get_input(P=np.array([[[1, 10, 0]]]),
    #             B=np.array([[[1, -2, 6]]]),
    #             Q=np.array([[[1, 0, 6]]]),
    #             T=repr.SimpleTotalTransitionFunc( repr.I(1) + repr.E(1,(1,2)) ))

    mock4 = get_input( CHC = [[ ["P"] , ["I"]] , [ ["I", "B", "T"] , ["I"] ], [ ["I", "~B" ] , ["Q"] ] ], 
            DNFs = {"P" : np.array([[[1, 10, 0]]]),
             "B" : np.array([[[1, -2, 6]]]),
             "Q" : np.array([[[1, 0, 6]]])
            }, 
            trans_funcs = {"T" : repr.SimpleTotalTransitionFunc( repr.I(1) + repr.E(1,(1,2)) )
            } , 
            invs = ["I"]
            )



