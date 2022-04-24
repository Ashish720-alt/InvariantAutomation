import pprint
import main
from guess import Guess, GuessStrategy
from configure import Configure as conf
from input import get_input
import numpy as np
import repr
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')


def get_surface_graph(program, conj_min, disj_min, mesh_range, mesh_step_size, runs, PRINT_DICTIONARY, random_strategy, max_const=None, guess_range=None):

    def compute_values():
        predicted_values = {}
        parameters = conf()
        parameters.max_disjuncts = disj_min
        parameters.max_conjuncts = conj_min
        cost = 0.0
        count_guess = 0
        while(parameters.max_disjuncts <= disj_min + mesh_range - 1):
            no_of_timeouts = 0
            count_sum = 0.0
            print("(", parameters.max_disjuncts, ",", parameters.max_conjuncts, "):")
            for k in range(runs):
                (_, cost, count_guess) = main.guess_inv(
                    program, parameters.max_guesses, random_strategy, max_const, guess_range)
                if (cost != 0):
                    no_of_timeouts = no_of_timeouts + 1
                count_sum = count_sum + (1.0 * count_guess)
            predicted_values[(parameters.max_disjuncts, parameters.max_conjuncts)] = (
                count_sum / (1.0 * runs), no_of_timeouts)
            parameters.max_conjuncts = parameters.max_conjuncts + mesh_step_size
            if (parameters.max_conjuncts > conj_min + mesh_range - 1):
                parameters.max_conjuncts = conj_min
                parameters.max_disjuncts = parameters.max_disjuncts + mesh_step_size
        if (PRINT_DICTIONARY):
            print("\n\n\n")
            pprint.pprint(predicted_values)
        return predicted_values

    def surfacegraphplotter(dict_of_values, conj_min, disj_min, mesh_range, mesh_step_size):
        fig = plt.figure(figsize=(15, 15))
        ax = plt.axes(projection='3d')
        T = []
        for v in dict_of_values.values():
            T.append(v[0])
        Z = np.array(T).reshape(mesh_range, mesh_range)
        x = np.arange(disj_min, disj_min + mesh_range, mesh_step_size)
        y = np.arange(conj_min, conj_min + mesh_range, mesh_step_size)
        X, Y = np.meshgrid(x, y)
        surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.cividis)

        # Set axes label
        ax.set_xlabel('Conjuncts', labelpad=20)
        ax.set_ylabel('Disjuncts', labelpad=20)
        ax.set_zlabel('Iterations', labelpad=20)
        fig.colorbar(surf, shrink=0.5, aspect=8)
        plt.show()

    values = compute_values()
    surfacegraphplotter(values, conj_min, disj_min, mesh_range, mesh_step_size)


program = get_input(P=np.array([[[1, 0, 0]]]),
                    B=np.array([[[1, -2, 6]]]),
                    Q=np.array([[[1, 0, 6]]]),
                    T=repr.SimpleTotalTransitionFunc(np.array([[1, 1], [0, 1]])))

get_surface_graph(program =program,conj_min= 1, disj_min = 1, mesh_range= 2, mesh_step_size= 1, runs= 1, PRINT_DICTIONARY= 1,
                random_strategy=GuessStrategy.SMALL_CONSTANT, max_const=10, guess_range=None)
# getSurfaceGraph(program, 1, 1, 1, 1, 1, 1, GuessStrategy.OCTAGONAL_DOMAIN )
# getSurfaceGraph(program, 1, 1, 1, 1, 1, 1, GuessStrategy.OCTAGONAL_DOMAIN_EXTENDED)
# getSurfaceGraph(program, 1, 1, 1, 1, 1, 1, GuessStrategy.NEAR_CONSTANT, guess_range = 1)
# getSurfaceGraph(program, 1, 1, 1, 1, 1, 1, GuessStrategy.NEAR_CONSTANT, guess_range = 5)
