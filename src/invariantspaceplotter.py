# First networkx library is imported 
# along with matplotlib
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mtcls
import numpy as np
import math
from configure import Configure as conf
import itertools as it
from cost_funcs import cost
from dnfs_and_transitions import list3D_to_listof2Darrays
import copy
from decimal import Decimal

def plotinvariantspace( maxconst, rotgraph, samplepoints, I_c, I_d, iteration_ct):
        
    def invariantencoding( I):
        def ccEncoding (cc):
            def pEncoding (p):
                return  p[:-2] + [p[-1]]
            rv = []
            for p in cc:
                rv = rv + pEncoding(p)
            return rv    
        
        rv = []
        for cc in I:
            rv = rv + ccEncoding(cc)
        return tuple(rv)

    
    def allInvs( maxconst, rotgraphKeys, I_c, I_d):    
        def allCCs (maxconst, rotgraphKeys, I_c):
            def allPredicates(maxconst, rotgraphKeys):
                rv = []
                for i in range(-maxconst, maxconst + 1):
                    for coeff in rotgraphKeys:
                        rv.append(list(coeff) + [-1, i])   
                return rv              
            allPreds = allPredicates(maxconst, rotgraphKeys)   

            allPredsTimesC = [allPreds for _ in range(I_c)]
            return  [list(v) for v in it.product(*allPredsTimesC)] 
        allccs = allCCs(maxconst, rotgraphKeys, I_c)
        allccsTimesD = [allccs for i in range(I_d)]
        return  [list(v) for v in it.product(*allccsTimesD)] 
    
    temp = allInvs(maxconst, rotgraph.keys(), I_c, I_d)
    V = {}
    costlist = []
    for (i,v) in enumerate(temp):
        V.update({ invariantencoding(v) : i })  
        costlist.append(cost(list3D_to_listof2Darrays(v), samplepoints)[0])
    V_ct = len(costlist)

    Eadj = {}
    for i in range(V_ct):
        Eadj[i] = []
        
    E = []
    edgecolor = []
    for v in temp:
        for (i,cc) in enumerate(v):
            for (j,p) in enumerate(cc):
                coeff = p[:-2]
                const = p[-1]                
                
                for coeff_n in rotgraph[ tuple(coeff) ]:
                    new_i = math.floor( i * (1.0 * np.dot(np.array(coeff_n), np.array(coeff)))/ (1.0 * np.dot(np.array(coeff), np.array(coeff)))  )
                    if (new_i > maxconst or new_i < - maxconst):
                        continue
                    p_new = coeff_n + [-1, new_i]
                    v_new = copy.deepcopy(v)
                    v[i][j] = p_new
                    
                    u1 = V[invariantencoding(v)]
                    u2 = V[invariantencoding(v_new)]
                    E.append( [ u1, u2] )
                    
                    Eadj[u1] = Eadj[u1] + [u2]
                    
                    edgecolor.append("green")
                    

                slope = math.sqrt(np.dot(np.array(coeff), np.array(coeff)))
                translation_range = math.floor(conf.translation_range * slope)
                translation_indices = list(range(-translation_range, translation_range +1))
                translation_indices.remove(0)
        
                for i_n in translation_indices:
                    const_n = const + i_n
                    if (const_n > maxconst or const_n < -maxconst):
                        continue
                    
                    p_new = coeff + [-1, const_n]
                    v_new = copy.deepcopy(v)
                    v[i][j] = p_new
                    u1 = V[invariantencoding(v)]
                    u2 = V[invariantencoding(v_new)]
                    E.append( [ u1, u2] )
                    
                    Eadj[u1] = Eadj[u1] + [u2]
                    edgecolor.append("black")  
    
            
    def BFS(Eadj, I_dist , s):
        visited = []
        queue = []
 
        queue.append(s)
        visited.append(s)
        I_dist[s] = 0
        while queue:
            t = queue.pop(0)
            d = I_dist[t]
            for i in Eadj[t]:
                if i not in visited:
                    I_dist[i] = d + 1
                    queue.append(i)
                    visited.append(i)        
        
        return



    Ig = -1
    for i in range(V_ct):
        if (costlist[i] == 0):
            Ig = i
            break
    if (Ig == -1):
        print("No ground truth here")
        return
    

    I_dist = [ 0 ] * V_ct
    BFS(Eadj, I_dist, Ig)

    d_max = max(I_dist)
    
    # print(V_ct, Ig, d_max)
    
    
    dist_dict = {}
    for i in range(d_max + 1):
        dist_dict[i] = 0
    for i in range(V_ct):
        dist_dict[I_dist[i]] = dist_dict[I_dist[i]] + 1
    for i in range(d_max+1):
        temp = dist_dict[i]
        dist_dict[i] = [temp, temp]
        
    def position(n, K, r):
        A = (2.0 * math.pi * K)/ n
        return np.array([ r * math.cos(A), r * math.sin(A) ])


    pos = {}
    for i in range(len(I_dist)):
        d = I_dist[i]
        [n,k] = dist_dict[d]
        pos[i] = position(n, k, d)
        dist_dict[d] = [n,k-1]
        
    labellist = {}
    for (i,a ) in enumerate(costlist):
        A = '%.1E' % Decimal(str(a))
        labellist.update({i : A })

    G = nx.Graph(E)

    
    # pos = nx.spring_layout(G, seed=3113794652)

    M = max(costlist)
    m = M
    for c in costlist:
        if (c < m and c != 0):
            m = c
    

    a = m / M
    colorslist = [ c / M for c in costlist  ]
    nx.draw_networkx_nodes(G, pos, nodelist = list(range(V_ct)), node_size=50, node_color=colorslist,  
                           cmap=mtcls.LinearSegmentedColormap.from_list( 'clrs', [ (0.0, 'blue') , (a, 'yellow'), ( 1.0,'red')], N=256, gamma=1.0 )  ) #Add pos here
    nx.draw_networkx_edges(G, pos, edge_color = edgecolor, alpha=0.5 )
    nx.draw_networkx_labels(G, pos, labellist, font_size=7, font_color="black")
    plt.show()
    
    
    # pts = nx.spring_layout(G)
    # pts_array = np.array([pt[1] for pt in sorted(list(pts.items()))])
    # plt.figure()
    # ax=plt.gca()
    # ax.plot(*pts_array.T, 'o')
    # for i,j in E:
    #     ax.plot(*pts_array[[i-1,j-1],:].T,'k')
    # # nx.draw(G,with_labels = True, alpha=0.8) #NEW FUNCTION
    # plt.show()

    
    # # 3D
    # pts_array_z = np.hstack((np.array([pt[1] for pt in sorted(list(pts.items()))]), np.array(costlist)[:,np.newaxis]))
    # fig=plt.figure()
    # ax=fig.add_subplot(projection='3d')
    # ax.plot(*pts_array_z.T, 'o')
    # for i,j in E:
    #     ax.plot(*pts_array_z[[i-1,j-1],:].T,'k')    
    # # nx.draw(G,with_labels = True, alpha=0.8) #NEW FUNCTION
    # plt.show()    
    
    G1 = nx.Graph()
    
    for i in range(V_ct):
        G1.add_node(i, X = pos[i][0], Y = pos[i][1], color = costlist[i], Label = '%.1E' % Decimal(str(costlist[i])))
    
    def edgecolor_to_float(ec):
        if ec == "green":
            return 0.1
        else:
            return 0.9
    
    for (i,e) in enumerate(E):
        w = edgecolor_to_float(edgecolor[i])
        G1.add_edge(e[0], e[1], weight = w)        
    nx.write_gexf(G1, "visualize.gexf" + str(iteration_ct))
    

                        
    return



  