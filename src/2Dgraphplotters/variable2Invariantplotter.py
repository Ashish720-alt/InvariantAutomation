import matplotlib.pyplot as plt
import numpy as np
import cdd
import math

################################## HyperParameters ##############################

LARGE_INT = 10000

LargeScale = 1
SmallScale = 0
scale = LargeScale



################################## Input ###################################
g_I = [np.array([ [1,2,-1,21], [-1,-2,-1,-21] , [0,-1,-1,-6] ]) ]
curr_I = [np.array([[1,-1,-1,-3] ,[-1,-2,-1,-10] , [1,0,-1,0]]  )]
pluspoints = [ [1.0, 10.0] , ]
minuspoints = [[8.0, 7.0] ,	[10000.0, 9999.0] ,	[10000.0, 7.0] ,	[6.0, 5.0] ,	[10000.0, 5.0] ,	[-9999.0, -10000.0] ,	
                [10000.0, -10000.0] ,	[-1, -2] ,	[-2, -3] ,[-3, -4] ,	[-4, -5] ,	[-5, -6] ,	[4, 1] ,	[6, 2] ,	[5, 3] ,	[4, 3] ,	[5, 4] ,]
ICEpairs = [([10000.0, 10000.0], [10002.0, 9999.0]) ,	([-10000.0, 10000.0], [-9998.0, 9999.0]) ,	([-10000.0, -10000.0], 
                    [-9998.0, -10001.0]) ,	([-11, -11], [-9, -12]) ,	([-12, -12], [-10, -13]) ,	([-13, -13], [-11, -14]) ,	
                ([-14, -14], [-12, -15]) ,	([-16, -15], [-14, -16]) ,	([2, 8], [4, 7]) ,	([0, 0], [2, -1]) ,	([-1, 1], [1, 0]) ,	
                    ([-2, 14], [0, 13]) ,	([-3, 16], [-1, 15]) ,	([5, 5], [7, 4]) ,	([4, 6], [6, 5]) ,	([3, 8], [5, 7]) ,	
                    ([2, 10], [4, 9]) ,	([1, 12], [3, 11]) ,	([4, 8], [6, 7]) ,	([-10, 1], [-8, 0]) ,	([-8, 2], [-6, 1]) ,	
                    ([-9, 1], [-7, 0]) ,	([5, 7], [7, 6]) ,	([4, 13], [6, 12]) ,	([-14, 3], [-12, 2]) ,	([-13, 4], [-11, 3]) ,	
                    ([-13, 3], [-11, 2]) ,	([4, 12], [6, 11]) ,	([6, 9], [8, 8]) ,	([5, 10], [7, 9]) ,	([6, 11], [8, 10]) ,	
                    ([6, 12], [8, 11]) ,	([6, 13], [8, 12])]


################################## Helper Functions ###################################

def get_centroid (points):
    x = 0.0
    y = 0.0
    n = 0.0
    for pt in points:
        x = x + pt[0]
        y = y + pt[1]
        n = n + 1.0
    return [ x/n, y/n ]


def get_listindex_from_list_of_lists(ll, k):
    rv = []
    for l in ll:
        rv.append(l[k])
    return rv



def deepcopy_DNF(I):
    n = len(I[0][0]) - 2    
    I_new = []
    for cc in I:
        cc_new = np.empty( shape=(0, n + 2), dtype = int )
        for p in cc:
            cc_new = np.concatenate((cc_new, np.array([np.copy(p)], ndmin=2)))
        I_new.append(cc_new)
    return I_new

def dnfconjunction (dnf1, dnf2):    
    ret = []
    for cc1 in dnf1:
        for cc2 in dnf2:
            cc = np.append(cc1, cc2, axis = 0)
            ret.append(cc)   
    return deepcopy_DNF(ret)

def Dstate(n):
    cc = np.empty(shape=(0,n+2), dtype = int)
    for i in range(n):
        p1 = np.zeros(n+2)
        p1[n] = -1
        p1[i] = -1
        p1[n+1] =  LARGE_INT

        p2 = np.zeros(n+2)
        p2[n] = -1
        p2[i] = 1
        p2[n+1] = LARGE_INT
        
        cc = np.concatenate((cc, np.array([p1, p2], ndmin=2)))
    
    return [cc]
        

# Assumes cc has LI predicates only, and not genLI predicates
def v_representation (cc):
    def pred_to_matrixrow (p):
        matrixrow = np.roll(p * -1, 1)[:-1]
        matrixrow[0] = matrixrow[0] * -1
        return matrixrow

    mat = []
    for p in cc:
        mat.append(pred_to_matrixrow(p))
    
    mat_cdd = cdd.Matrix( mat, number_type='float')
    mat_cdd.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat_cdd)
    
    v_repr = poly.get_generators()
    tuple_of_tuple_generators = v_repr.__getitem__(slice(v_repr.row_size))
    list_of_list_generators = []
    for tuple_generator in tuple_of_tuple_generators:
        list_of_list_generators.append(list(tuple_generator)[1:])

    return list_of_list_generators






################################## Data Points and Scale ##############################

def takescale( scaleValue ):
    ax = plt.gca()
    if (scaleValue == LargeScale): #Large Scale
        ax.set_xlim([-LARGE_INT-250,LARGE_INT+250])
        ax.set_ylim([-LARGE_INT-250,LARGE_INT+250])
    else: # Small Scale
        ax.set_xlim([-50,50])
        ax.set_ylim([-50,50])

def plotpoints(pluspoints, minuspoints, ICEpairs):
    #positive points
    for pt in pluspoints:
        plt.plot([pt[0]], [pt[1]], marker="o", markersize=10, color="green")

    #Negative Points
    for pts in minuspoints:
        plt.plot([pts[0]], [pts[1]], marker="x", markersize=10, color= "red")


    #ICE pairs
    for pair in ICEpairs:
        hdx = pair[0][0]
        hdy = pair[0][1]
        changex = pair[1][0] - pair[0][0]
        changey = pair[1][1] - pair[0][1]
        plt.arrow(hdx, hdy, changex, changey, head_width = 0.4, color = "blue")


################################## Plot DNF ######################################

#Assumes DNF in LII form and not gLII form
def plotDNF(DNF, dashstyle, boundarycolor, fillcolor):
    DNF_bounded = dnfconjunction( DNF , Dstate(2))

    for i in range(len(DNF_bounded)):    
        pts_S = v_representation( np.array(DNF_bounded[i], ndmin = 2) )
        c_S = get_centroid(pts_S)

        def clockwiseangle_and_distance1( point):
            origin = c_S
            refvec = [0, 1]
            # Vector between point and the origin: v = p - o
            vector = [point[0]-origin[0], point[1]-origin[1]]
            # Length of vector: ||v||
            lenvector = math.hypot(vector[0], vector[1])
            # If length is zero there is no angle
            if lenvector == 0:
                return -math.pi, 0
            # Normalize vector: v/||v||
            normalized = [vector[0]/lenvector, vector[1]/lenvector]
            dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
            diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
            angle = math.atan2(diffprod, dotprod)
            # Negative angles represent counter-clockwise angles so we need to subtract them 
            # from 2*pi (360 degrees)
            if angle < 0:
                return 2*math.pi+angle, lenvector
            # I return first the angle because that's the primary sorting criterium
            # but if two vectors have the same angle then the shorter distance should come first.
            return angle, lenvector

        S = sorted( pts_S  , key=clockwiseangle_and_distance1)
        plt.plot(get_listindex_from_list_of_lists(S, 0), get_listindex_from_list_of_lists(S, 1), dashstyle + boundarycolor)
        plt.fill( [pt[0] for pt in S] , [pt[1] for pt in S], fillcolor)


def do_plot(plotname, scale, currentInv, groundtruth = []):
    takescale(scale)
    if (groundtruth != []):
        plotDNF(groundtruth, '-' , 'm', 'tab:pink')
    plotDNF(currentInv, '-' , 'k', '0.8')
    plotpoints(pluspoints, minuspoints, ICEpairs)
    plt.title(plotname)
    plt.xlabel('x', color='#1C2833')
    plt.ylabel('y', color='#1C2833')
    plt.grid()
    plt.savefig(plotname)
    plt.show()
    

do_plot('Invariant Plot' + '(LargeScale)', LargeScale, curr_I, g_I)
do_plot('Invariant Plot' + '(SmallScale)', SmallScale, curr_I, g_I)

#Dotted Lines
# dashstylelist = [5,5,5,5]
# def const_list(k):
#     return [k,k,k]
# x = np.linspace(-LARGE_INT,LARGE_INT,LARGE_INT)
# y1 = (21-x)/2
# ax1 = plt.plot(x, y1, '-g',dashes=dashstylelist )
# y2 = [6,6,6]
# ax1 = plt.plot([-LARGE_INT,0,LARGE_INT], y2, '-g',dashes=dashstylelist )

# Dotted Lines
# x = np.linspace(-LARGE_INT,LARGE_INT,LARGE_INT)
# y1 = x+3
# y2 = (10-x)/2
# y3 = [-LARGE_INT,0,LARGE_INT]
# plt.plot(x, y1, '-b',dashes=dashstylelist )
# plt.plot(x, y2, '-b', dashes=dashstylelist )
# plt.plot([0,0,0], y3, '-b', dashes=dashstylelist)


# ax1 = plt.plot(x, y1, '-b', label='2x-y+1=0')
# # ax1.fill_between(x, LARGE_INT, y1)
# ax1 = plt.plot(x, y2, '-b', label='x-y-1=0')
# # ax1.fill_between(x, -LARGE_INT, y2)
# ax1 = plt.plot(const_list(3), y3 , '-b', label='x = 3')
# # ax1.fill_between(x, LARGE_INT, y1)


