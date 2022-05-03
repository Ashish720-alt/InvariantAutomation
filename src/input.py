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


def get_input(P, B, Q, T):
    return repr.Repr(P, B, Q, T)

class mock:
    mock1 = get_input(P=np.array([[[1, 0, 0, 0, 1], [0, 0, 1, 0, 0], [0, 1, 0, 1, 1], [0, 1, 0, -1, 1000]]]),
                    B=np.array([[[1, -1, 0, -2, 0]]]),
                    Q=np.array([[[0, -1, 1, 1, 0]]]),
                    # now we directly call it from repr, ideally we should do it in get_input(C_source_code)
                    T=repr.SimpleTotalTransitionFunc(np.array([[1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]])))

    mock2 = get_input(P=np.array([[[1, 0, 0]]]),
                    B=np.array([[[1, -2, 6]]]),
                    Q=np.array([[[1, 0, 6]]]),
                    T=repr.SimpleTotalTransitionFunc(np.array([[1, 1], [0, 1]])))


