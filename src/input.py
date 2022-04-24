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
