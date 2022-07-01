import pprint
from configure import Configure as conf
import input 
import numpy as np
import repr
from cost_funcs import Cost
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')


def get_surface_graph_cost_funs(program, coeff_min, const_min, mesh_range, mesh_step_size, PRINT_DICTIONARY):

    def compute_values():
        predicted_values = {}
        parameters = conf()
        parameters.max_disjuncts = 1
        parameters.max_conjuncts = 1

        coeff = coeff_min
        const = const_min

        while(coeff < coeff_min + mesh_range):
            I = np.array([[[coeff, -1, const]]])        
            predicted_values[(coeff, const)] =  -Cost(program, I).get_cost() 
            print(coeff, const, predicted_values[(coeff, const)])
            if (const < const_min + mesh_range - mesh_step_size):
                const = const + mesh_step_size
            else:
                const = const_min
                coeff = coeff + mesh_step_size     
        
        if (PRINT_DICTIONARY):
            print("\n\n\n")
            pprint.pprint(predicted_values)        


        return predicted_values

    def surfacegraphplotter(dict_of_values, coeff_min, const_min, mesh_range, mesh_step_size):
        fig = plt.figure(figsize=(15, 15))
        ax = plt.axes(projection='3d')
        T = []
        for v in dict_of_values.values():
            T.append(v)
        Z = np.array(T).reshape(mesh_range, mesh_range)
        x = np.arange(coeff_min, coeff_min + mesh_range, mesh_step_size)
        y = np.arange(const_min, const_min + mesh_range, mesh_step_size)
        X, Y = np.meshgrid(x, y)
        surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.cividis)

        # Set axes label
        ax.set_xlabel('Coefficients', labelpad=20)
        ax.set_ylabel('Constants', labelpad=20)
        ax.set_zlabel('-Cost', labelpad=20)
        fig.colorbar(surf, shrink=0.5, aspect=8)
        plt.show()

    values = compute_values()
    surfacegraphplotter(values, coeff_min, const_min, mesh_range, mesh_step_size)


program = input.mock.mock2

get_surface_graph_cost_funs(program = program, coeff_min = -6, const_min = 6, mesh_range = 12, mesh_step_size = 1, PRINT_DICTIONARY = 1)

