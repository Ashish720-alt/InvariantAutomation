import matplotlib.pyplot as plt
import numpy as np
import math
from dnfs_and_transitions import dnfconjunction, dnfdisjunction, dnfnegation
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




def do_UUplot(plotname, foldername, scale, pluspoints, minuspoints, unknownimplication, show = False, resolution = conf.n2PLOTTER_LOW_RES):
    #Turn off Interactive Mode, otherwise each time you save, it shows plot.
    plt.ioff()
    plt.grid()
    takescale(scale)

    for pt in unknownimplication:
        plt.plot([pt[0]], [pt[1]], marker="D", markersize=1, color="lightsteelblue")

    for pt in minuspoints:
        plt.plot([pt[0]], [pt[1]], marker="_", markersize=2, color="red")

    for pt in pluspoints:
        plt.plot([pt[0]], [pt[1]], marker="+", markersize=4, color="darkgreen")



    plt.title(plotname)
    plt.xlabel('x', color='#1C2833')
    plt.ylabel('y', color='#1C2833')
    plt.savefig(foldername + '/' + plotname,  dpi= resolution)
    if (not show):
        plt.close()
    return



def do_CUplot(plotname, foldername, scale, pluspoints, minuspoints, neutralpoints, show = False, resolution = conf.n2PLOTTER_LOW_RES):
    #Turn off Interactive Mode, otherwise each time you save, it shows plot.
    plt.ioff()
    plt.grid()
    takescale(scale)

    for pt in neutralpoints:
        plt.plot([pt[0]], [pt[1]], marker="o", markersize=1, color="yellow")

    for pt in minuspoints:
        plt.plot([pt[0]], [pt[1]], marker="_", markersize=2, color="red")

    for pt in pluspoints:
        plt.plot([pt[0]], [pt[1]], marker="+", markersize=4, color="green")



    plt.title(plotname)
    plt.xlabel('x', color='#1C2833')
    plt.ylabel('y', color='#1C2833')
    plt.savefig(foldername + '/' + plotname,  dpi= resolution)
    if (not show):
        plt.close()
    return


p = [[1,1]]

n = []
for x in range(-50, 51):
    for y in range(-50, 1):
            n.append([x,y])   

u = []
for x in range(-50, 51):
    for y in range(-50, 51):
        if ([x,y] in p or [x,y] in n):
            continue
        u.append([x,y])

do_UUplot('0U-partition', '2DInvariantPlots', conf.n2PLOTTER_SMALLSCALE, p, n, u )

# Tp = [ ]
# curr = [1, 1]
# while ((curr[0] < 50 and curr[0] > -50) and (curr[1] < 50 and curr[1] > -50)):
#     Tp.append(curr)
#     temp = curr[0] + curr[1]
#     curr = [temp, temp]
    
# Tm = [ ]
# for x in range(-50, 51):
#     for y in range(-50, 51):
#         if (y <= 0 or (x + y <= 0)):
#             Tm.append([x,y])    

# Tn = []    
# for x in range(-50, 51):
#     for y in range(-50, 51):
#         if ([x,y] in Tp or [x,y] in Tm):
#             continue
#         Tn.append([x,y])
        
# do_CUplot('CU-partition', '2DInvariantPlots', conf.n2PLOTTER_SMALLSCALE, Tp, Tm, Tn )
