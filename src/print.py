from configure import Configure as conf

def DNF_aslist(I):
    I_list = []
    for cc in I:
        cc_list = []
        for p in cc:
            cc_list.append(list(p))
        I_list.append(cc)
    return I_list

def initialized():
    if (conf.PRINT_ITERATIONS == conf.ON):
        print("Initialization Complete...")

def statistics(t, I, cost, mincost):
    I_list = DNF_aslist(I)
    if (conf.PRINT_ITERATIONS == conf.ON):
        print("t = ", t, "\t", I_list , "\t", "(cost, mincost) = ", cost, mincost)

def z3statistics(correct, original_samplepoints, added_samplepoints):
    if (conf.PRINT_ITERATIONS == conf.ON):    
        print("Z3 Statistics:\n", "correct = ", correct, "\n", "original-selection-points:\n", original_samplepoints, "\n", "CEX-generated:\n", added_samplepoints )

