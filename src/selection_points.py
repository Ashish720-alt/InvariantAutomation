import repr
import numpy as np
import cdd


def v_representation (cc):
    
    def pred_to_matrixrow (p):
        return 1
    
    mat = cdd.Matrix([[0, 0, 1],[1, -1, 0],[0, 1, -1]], number_type='fraction')
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)
    
    v_repr = poly.get_generators()
    tuple_of_tuple_generators = v_repr.__getitem__(slice(v_repr.row_size))
    list_of_list_generators = []
    for tuple_generator in tuple_of_tuple_generators:
        list_of_list_generators.append(list(tuple_generator)[1:])
    print(list_of_list_generators)

    return list_of_list_generators

def get_positive (P, Dstate):
 return 1


v_representation(1)