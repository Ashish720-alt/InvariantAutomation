import matplotlib.pyplot as plt
import numpy as np
import math
from dnfs_and_transitions import dnfconjunction
from selection_points import Dstate, v_representation
from configure import Configure as conf


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


################################## Data Points and Scale ##############################

def takescale( scaleValue ):
    ax = plt.gca()
    if (scaleValue == conf.n2PLOTTER_CUSTOMIZEDSCALE): #Customized Scale
        ax.set_xlim(conf.n2PLOTTER_CUSTOMIZEDSCALEINTERVAL)
        ax.set_ylim(conf.n2PLOTTER_CUSTOMIZEDSCALEINTERVAL)
    elif (scaleValue == conf.n2PLOTTER_LARGESCALE): #Large Scale
        ax.set_xlim(conf.n2PLOTTER_LARGESCALEINTERVAL)
        ax.set_ylim(conf.n2PLOTTER_LARGESCALEINTERVAL)
    else: # Small Scale
        ax.set_xlim(conf.n2PLOTTER_SMALLSCALEINTERVAL)
        ax.set_ylim(conf.n2PLOTTER_SMALLSCALEINTERVAL)



def plotpoints(pluspoints, minuspoints, ICEpairs):
    #positive points
    for pt in pluspoints:
        plt.plot([pt[0]], [pt[1]], marker="+", markersize=4, color="green")

    #Negative Points
    for pts in minuspoints:
        plt.plot([pts[0]], [pts[1]], marker="_", markersize=4, color= "red")


    #ICE pairs
    for pair in ICEpairs:
        hdx = pair[0][0]
        hdy = pair[0][1]
        changex = pair[1][0] - pair[0][0]
        changey = pair[1][1] - pair[0][1]
        plt.arrow(hdx, hdy, changex, changey, head_width = 0.2, color = "blue")


################################## Plot DNF ######################################


def plotDNF(DNF, dashstyle, boundarycolor, transparency):
    DNF_bounded = dnfconjunction( DNF , Dstate(2), 0)

    for i in range(len(DNF_bounded)):    
        pts_S = v_representation( np.array(DNF_bounded[i], ndmin = 2) )
        if (len(pts_S) == 0):
            continue
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
        S = S + [S[0]]
        plt.plot(get_listindex_from_list_of_lists(S, 0), get_listindex_from_list_of_lists(S, 1), dashstyle + boundarycolor)
        plt.fill( [pt[0] for pt in S] , [pt[1] for pt in S], color = boundarycolor, alpha = transparency)


def do_plot(plotname, foldername, scale, I, samplepoints, show = False, resolution = conf.n2PLOTTER_LOW_RES, Ig = []):
    #Turn off Interactive Mode, otherwise each time you save, it shows plot.
    plt.ioff()
    plt.grid()
    takescale(scale)
    if (Ig != []):
        plotDNF(Ig, '-' , 'm', 0.4)
    plotDNF(I, '-' , 'k', 0.4)

    
    plotpoints(samplepoints[0], samplepoints[1], samplepoints[2])
    plt.title(plotname)
    plt.xlabel('x', color='#1C2833')
    plt.ylabel('y', color='#1C2833')
    plt.savefig(foldername + '/' + plotname,  dpi=1000)
    if (not show):
        plt.close()
    return



# g_I = [np.array([ [1,2,-1,21], [-1,-2,-1,-21] , [0,-1,-1,-6] ]) ]
# curr_I = [np.array([[-1,-1,-1,-27] ,[1,2,-1,-24] ]  )]
# pluspoints = [ [1.0, 10.0] , ]
# minuspoints = [[8.0, 7.0] ,	[10000.0, 9999.0] ,	[10000.0, 7.0] ,	[6.0, 5.0] ,	[10000.0, 5.0] ,	[-9999.0, -10000.0] ,	
#                 [10000.0, -10000.0] ,	[-1, -2] ,	[-2, -3] ,[-3, -4] ,	[-4, -5] ,	[-5, -6] ,	[4, 1] ,	[6, 2] ,	[5, 3] ,	[4, 3] ,	[5, 4] ,]
# ICEpairs = [([10000.0, 10000.0], [10002.0, 9999.0]) ,	([-10000.0, 10000.0], [-9998.0, 9999.0]) ,	([-10000.0, -10000.0], 
#                     [-9998.0, -10001.0]) ,	([-11, -11], [-9, -12]) ,	([-12, -12], [-10, -13]) ,	([-13, -13], [-11, -14]) ,	
#                 ([-14, -14], [-12, -15]) ,	([-16, -15], [-14, -16]) ,	([2, 8], [4, 7]) ,	([0, 0], [2, -1]) ,	([-1, 1], [1, 0]) ,	
#                     ([-2, 14], [0, 13]) ,	([-3, 16], [-1, 15]) ,	([5, 5], [7, 4]) ,	([4, 6], [6, 5]) ,	([3, 8], [5, 7]) ,	
#                     ([2, 10], [4, 9]) ,	([1, 12], [3, 11]) ,	([4, 8], [6, 7]) ,	([-10, 1], [-8, 0]) ,	([-8, 2], [-6, 1]) ,	
#                     ([-9, 1], [-7, 0]) ,	([5, 7], [7, 6]) ,	([4, 13], [6, 12]) ,	([-14, 3], [-12, 2]) ,	([-13, 4], [-11, 3]) ,	
#                     ([-13, 3], [-11, 2]) ,	([4, 12], [6, 11]) ,	([6, 9], [8, 8]) ,	([5, 10], [7, 9]) ,	([6, 11], [8, 10]) ,	
#                     ([6, 12], [8, 11]) ,	([6, 13], [8, 12])]
# do_plot('Invariant Plot' + '(LargeScale)', '2DInvariantPlots', conf.n2PLOTTER_LARGESCALE, curr_I, curr_I, curr_I, curr_I, (pluspoints, minuspoints, ICEpairs),  Ig = g_I)
# do_plot('Invariant Plot' + '(SmallScale)', '2DInvariantPlots', conf.n2PLOTTER_SMALLSCALE, curr_I, curr_I, curr_I, curr_I, (pluspoints, minuspoints, ICEpairs), 
# resolution = conf.n2PLOTTER_HIGH_RES, Ig = g_I)




