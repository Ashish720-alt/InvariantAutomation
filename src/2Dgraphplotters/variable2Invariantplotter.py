import matplotlib.pyplot as plt
import numpy as np
LARGE_INT = 10000

x = np.linspace(-LARGE_INT,LARGE_INT,2*(LARGE_INT))

#positive points
plt.plot([4], [3], marker="o", markersize=6, markerfacecolor="green")

#Negative Points
plt.plot([100], [2000], marker="o", markersize=6, markerfacecolor="red")

#ICE pairs
plt.quiver([0], [0], [1], [-1], color=["black"])



plt.plot(x, 2*x+1, '-b', label='2x-y+1=0')
plt.plot(x, x - 1, '-b', label='x-y-1=0')


plt.title('Invariant of Programm')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()