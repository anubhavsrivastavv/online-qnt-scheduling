import numpy as np
import networkx as nx
import math
import colors
import sys
import time
import random

###In this N -> number of requests (k), len(E) -> number of edges (packing constraints)

start = time.time()
# scale, epsilon = 1, 0.1
scale, epsilon = 1, 1
alpha, U = 1, 1   #U=0, doesn't work -> division by 0 in r

# Create the graph and demand set 
#Test Case 1
v = 9
N, T = 20, 5
E = [(0, 1), (1, 2), (0, 4), (1, 3), (2, 5), (4, 3), (3, 5), (4, 6), (3, 7), (5, 8), (6, 7), (7, 8)]
C = [2, 3, 3, 2, 3, 4, 2, 2, 1, 3, 2, 5]  #(edge id, capacity)
# D =    [(0, 2, 2, 1, 3),
#         (0, 3, 2, 2, 3),
#         (0, 5, 3, 1, 4),
#         (2, 7, 2, 3, 4),
#         (3, 6, 3, 4, 5)]

# D =    [(0, 2, 2, 1, 3),
#         (0, 3, 5, 2, 3),
#         (0, 5, 2, 1, 4),
#         (2, 7, 5, 3, 4),
#         (3, 6, 4, 4, 5),
#         (1, 3, 4, 1, 2),
#         (4, 7, 2, 2, 5),
#         (2, 7, 3, 3, 4)]
# D = [(8, 7, 2, 0, 1), (4, 3, 3, 2, 4), (6, 8, 3, 3, 4), (6, 3, 3, 3, 4), (1, 7, 4, 0, 3)]

#Test Case 2
# v = 4
# E = [(0, 1), (1, 2), (1, 3), (2, 3)]
# C = [2, 2, 2, 2] 
# N, T = 2, 3
# D = [(0, 3, 4, 1, 3), (0, 2, 3, 1, 3)] #(s, e, d, ST, ET)

#Methdod to generate demands
def generate_demands(nb_demands, nb_nodes, nb_timesteps, max_eprs, debug=False):
    demand_sizes = np.random.randint(1, max_eprs, size=nb_demands)

    all_pairs = []
    for i in range(nb_nodes):
        for j in range(nb_nodes):
            if i != j:
                all_pairs.append((i, j))
    demand_pairs = random.choices(all_pairs, k=nb_demands)
    start_nodes, end_nodes = list(zip(*demand_pairs))

    all_intervals = []
    for i in range(nb_timesteps):
        for j in range(i + 1, nb_timesteps):
            all_intervals.append((i, j))
    demand_intervals = random.choices(all_intervals, k=nb_demands)
    start_times, end_times = list(zip(*demand_intervals))
    # start_times = list(start_times)
    # end_times = list(end_times)
    print(start_times)
    print(end_times)

    return start_nodes, end_nodes, demand_sizes, start_times, end_times

def demands_to_algo(start_nodes, end_nodes, D, DST, DT):
    D_matrix = []
    for i in range(N):
        D_matrix.append((start_nodes[i], end_nodes[i], D[i], DST[i], DT[i]))
    return D_matrix
    

start_nodes, end_nodes, D_size, DST, DT = generate_demands(nb_demands=N, nb_nodes=v, nb_timesteps=T, max_eprs=5)
D = demands_to_algo(start_nodes, end_nodes, D_size, DST, DT)
print(D)

E_dict = {}
for idx, e in enumerate(E):
    E_dict[e] = idx

G = nx.Graph(E)
print(G)
nx.draw_networkx(G)

# Find the candidate paths for all demands
P = {}
edge_path_map = {}
for i in range(N):
    all_p = []
    for p in nx.all_simple_paths(G, D[i][0], D[i][1], 3):
        print(f'demand: {i} path: {p}')
        p_idx = []
        for j in range(len(p)-1):
            e_idx = -1
            if (p[j], p[j+1]) in E_dict:
                e_idx = E_dict[(p[j], p[j+1])]
            else:
                e_idx = E_dict[(p[j+1], p[j])]
            p_idx.append(e_idx)
            
            #Store all paths covering an edge
            if e_idx not in edge_path_map:
                edge_path_map[e_idx] = [(i, len(all_p))] #len(all_p) points to number of paths stored for req i, therefore it is path index
            else:
                edge_path_map[e_idx].append((i, len(all_p)))
        all_p.append(p_idx)
    P[i] = all_p

print(P)

# scale = np.ceil(np.log((len(C)*(T+1)+len(D))*(len(D)**(1+epsilon)/(1-(epsilon/2)))))
# print(scale)

# scale = np.log(len(E)*T*N) 

scale = 0.8
print(scale)

# Set u_{i,p,t} values
u = {}
for i in range(N):
    for idx, p in enumerate(P[i]):
        for t in range(D[i][3], D[i][4]+1):
            u[f'u_{i}_{idx}_{t}'] = 1/(N*D[i][2])
            print(u)

#Flow variables:
F = {}
for i in range(N):
    for idx, p in enumerate(P[i]):
        for t in range(D[i][3], D[i][4]+1):
            F[f'f_{i}_{idx}_{t}'] = 0

#h_{e,t} ---> Not necessarily needed as Y_et is updated 
H = {}
for e in range(len(E)):
    for t in range(T+1):
        H[f'h_{e}_{t}'] = 0

#Capacity Usage --- (1)
# for i in range(N):
#     for t in range(D[i][3], D[i][4]+1):
#         for idx, p in enumerate(P[i]):
#             for e in p:
#                 H[f'h_{e}_{t}'] += F[f'f_{i}_{idx}_{t}']

#L_i
L = {}
for i in range(N):
    L[f'l_{i}'] = 0

#Demand Satisfied  --- (2)
def compute_li():
    for i in range(N):
        L[f'l_{i}'] = 0
        for t in range(D[i][3], D[i][4]+1):
            for idx, p in enumerate(P[i]):
                L[f'l_{i}'] += F[f'f_{i}_{idx}_{t}']

def li(i):
    l_i= 0
    for t in range(D[i][3], D[i][4]+1):
        for idx, p in enumerate(P[i]):
            l_i += F[f'f_{i}_{idx}_{t}']
    return l_i

#Utility --- (3):
g = 0
def compute_g(g):
    g = 0
    for i in range(N):
        for idx, p in enumerate(P[i]):
            for t in range(D[i][3], D[i][4]+1):
                g += u[f'u_{i}_{idx}_{t}']*F[f'f_{i}_{idx}_{t}']
    return g


"""
Variables initialization
"""
#y_{e,t}
Y = {}
for e in range(len(E)):
    for t in range(T+1):
        Y[f'y_{e}_{t}'] = np.exp((scale*H[f'h_{e}_{t}'])/(epsilon*C[e]))

print(Y)
#q_i
Q = {}
for i in range(N):
    Q[f'q_{i}'] = np.exp((scale*L[f'l_{i}'])/(epsilon*D[i][2]))
print(Q)

#z_i
Z = {}
for i in range(N):
    Z[f'z_{i}'] = np.exp(-(scale*L[f'l_{i}'])/(epsilon*alpha*D[i][2]))
print(Z)

r = np.exp(-(scale*g)/(epsilon*U))
print(r)

"""
    Calendering paper implementation
"""
def should_loop():
    for i in range(N):
        print(colors.yellow + f'inside should loop alpha[{i}], L[l_{i}], alpha[{i}]*D[i][2], U, g: {alpha[i], L[f"l_{i}"], alpha[i]*D[i][2], U, g}')
        # if L[f'l_{i}'] < alpha[i]*D[i][2] or g < U:
        # l_i = li(i)
        # print(l_i)
        # if  l_i < alpha[i]*D[i][2]:
        if L[f'l_{i}'] < alpha[i]*D[i][2]:
            print(colors.bright_white + f'yes loop, req_details: {D[i]}')
            return True
    print(colors.bright_white + 'no - dont loop')
    return False

def find_req():
    for i in range(N):
        for t in range(D[i][3], D[i][4]+1):
            for idx, p in enumerate(P[i]):
                print(colors.bright_white + f'{i, t, idx}')
                print(colors.pure_red+f'demand: {D[i]} and time: {t}')
                print(colors.dark_green +f'(edge index, edge, capacity) :{[(e, E[e], C[e]) for e in p]}')
                print(f'Edges: {E}')
                print(f'Capacity: {C}')
                print(f'Y: {Y}')
                print(f'Q: {Q}')
                left_side_num = sum([Y[f'y_{e}_{t}']/C[e] for e in p])+Q[f'q_{i}']/D[i][2]
                left_side_den = sum(Y.values()) + sum(Q.values())

                # print(f'left_side_den: {left_side_den}')
                # left_side_den = 0
                # for e in range(len(E)):
                #     for t_prime in range(T+1):
                #         left_side_den += Y[f'y_{e}_{t_prime}']
                # for j in range(N):
                #     left_side_den += Q[f'q_{j}']
                # print(f'left_side_den: {left_side_den}')
                print(colors.light_green)
                print(f'Req {i, (D[i])} Time: {t} and Path:{[E[e] for e in p]}')
                print(colors.bold + f'left num: {left_side_num} den:{left_side_den}')

                right_side_num = 0
                l_i = li(i)
                print(colors.yellow + f'Z: {Z}')
                print(colors.bright_white)
                print(f'l_i: {l_i} and alpha[i]*D[i][2]: {alpha[i]*D[i][2]}')
                if L[f'l_{i}'] < alpha[i]*D[i][2]:
                # if l_i < alpha[i]*D[i][2]:
                    right_side_num += Z[f'z_{i}']/(alpha[i]*D[i][2])
                # if g < U:
                #     right_side_num += u[f'u_{i}_{idx}_{t}']*r/U

                right_side_den = 0
                for j in range(N):
                    if L[f'l_{j}'] < alpha[j]*D[j][2]:
                        # print(f'added req: {D[j]}')
                        right_side_den+= Z[f'z_{j}']
                    else:
                        print(f'skipped req: {D[j]}')
                # if g < U:
                #     right_side_den+= r
                
                print(colors.bold + f'right num: {right_side_num} den:{right_side_den}')
                print(colors.bold + f'left val: {left_side_num/left_side_den} right val: {right_side_num/right_side_den}')
                if left_side_num/left_side_den <= right_side_num/right_side_den:
                    # print(f'{i, t, idx}')
                    print(colors.bright_white + 'found')
                    return (i, t, idx)
    print(colors.bright_white)
    return None

iter = 0
# max_D = max([D[i][2] for i in range(len(D))])
for a in np.arange(0.1, 1.1, 0.1):
    alpha = [a for i in range(len(D))]
# for nEPR in range(1, max_D+1, 1):
    # alpha = [nEPR/D[i][2] if nEPR<D[i][2] else 1 for i in range(len(D))]
    print(f'computed alpha: {alpha}')
    isInfeasible = False
    while should_loop():
        iter += 1
        print(f'alpha: ------ {alpha} and iteration : {iter}')
        request = find_req()
        # break

        if request == None:
            print(f'could not allocate at iteration: {iter}')
            break

        i_star, t_star, p_star = request
        print(i_star, t_star, p_star)
        min_ = math.inf  #Finding min edge capacity along the path p_star
        for e in P[i_star][p_star]:
            # print(P[i_star][p_star])
            e_capacity = C[e]
            if e_capacity < min_:
                min_ = e_capacity

        gamma = epsilon*min(D[i_star][2], min_)
        print(f'gamma: {gamma}')

        if L[f'l_{i_star}'] < alpha[i_star]*D[i_star][2]:
            gamma = min(gamma, epsilon*alpha[i_star]*D[i_star][2])
        
        print(f'gamma: {gamma}')

        # if g < U:
        #     gamma = min(gamma, epsilon*U/u[f'u_{i_star}_{p_star}_{t_star}'])
        
        print(f'gamma: {gamma}')
        
        for e in P[i_star][p_star]:
            print(f"Bef updating: Y[f'y_{e}_{t_star}'] : {Y[f'y_{e}_{t_star}']}")
            Y[f'y_{e}_{t_star}'] = Y[f'y_{e}_{t_star}']*np.exp(gamma/C[e])
            print(f"Aft updating: Y[f'y_{e}_{t_star}'] : {Y[f'y_{e}_{t_star}']}")     
        
        
        Q[f'q_{i_star}'] = Q[f'q_{i_star}']*np.exp(gamma/D[i_star][2])

        Z[f'z_{i_star}'] = Z[f'z_{i_star}']*np.exp(-gamma/(alpha[i_star]*D[i_star][2]))

        r = r*np.exp((-u[f'u_{i_star}_{p_star}_{t_star}']*gamma)/U)

        F[f'f_{i_star}_{p_star}_{t_star}'] = F[f'f_{i_star}_{p_star}_{t_star}'] + (epsilon/scale)*gamma
        print(f'f_{i_star}_{p_star}_{t_star} updated by  {(epsilon/scale)*gamma}')
    
        #End of While
        compute_li()
        g = compute_g(g)
        print(F)
        print(g)
    
    # break

    if isInfeasible:
        print(f'max alpha: {[a - 0.1 for a in alpha]}')
        break
"""
    End Calendering Paper implementation
"""


"""
    Begin Post Processing and Edge capacity checks
"""
t_begin_post = time.time()
#We have a prelimnary value of f now:
print(colors.bright_blue+'Computed f by Youngs method')
print(F)

for f, v in F.items():
    F[f] = round(v)

print(colors.light_green + 'Clipped F by rounding')
print(F)   #Clipped values of f

#Heursitic to correct the edge usage and adjust allocations
alpha = [li(i)/D[i][2] for i in range(N)]
print(alpha)

def get_edge_allocations():
    edge_alloc = {}
    for e in range(len(E)):
        for t in range(T+1):
            edge_alloc[f'e_{e}_{t}'] = 0

    #Find the total allocation on an edge at time t
    for i in range(N):
        for idx, p in enumerate(P[i]):
            for e in p:
                for t in range(D[i][3], D[i][4]+1):
                    edge_alloc[f'e_{e}_{t}'] += F[f'f_{i}_{idx}_{t}']
    return edge_alloc

edge_allocation = get_edge_allocations()
print(colors.bright_white + 'The edge allocations are given below')
print(edge_allocation)

#Find the edges where allocation exceeds capacity
def get_conflicts(edge_alloc):
    conflicts = {} 
    for e in range(len(E)):
        for t in range(T+1):
            if edge_alloc[f'e_{e}_{t}'] > C[e]:
                #Find the paths that flow through this edge and 
                #contribute to the flow on edge 'e' at time 't'
                p_lst = []
                for (i, p_idx) in edge_path_map[e]:
                    if t>=D[i][3] and t<=D[i][4] and F[f'f_{i}_{p_idx}_{t}'] > 0:
                        p_lst.append((p_idx, i, alpha[idx]))
                conflicts[(e, t)] = p_lst
    return conflicts

conflicts_ = get_conflicts(edge_allocation)
print(colors.pure_red+'detected conflicts')
print(conflicts_)

def remove_flow(conflicts):
    #Sort the flows along each conflicted edge in descending order of alpha(fraction completed)
    for e, paths in conflicts.items():
        paths.sort(key=lambda tup: tup[2])

    print(colors.light_green+'Sorted Conflicts')
    print(conflicts)
    #Remove the flow with max alpha
    for edge_time, paths in conflicts.items():
        _ , t = edge_time
        p_idx, i, alpha_i = paths[0]
        F[f'f_{i}_{p_idx}_{t}'] = 0
        del paths[0]

    return conflicts

#Recompute the conflicts, while no conflicts remain repeat
while len(conflicts_.keys()) > 0:
    conflicts_ = remove_flow(conflicts_)
    print(colors.yellow + f'After removing flow with max alpha \nUpdated F:')
    print(F)

    print(colors.dark_green + f'Conflicts after removing flow')
    print(conflicts_)
    
    edge_allocation = get_edge_allocations()
    print(colors.bright_white + f'Recomputed Edge Allocations')
    print(edge_allocation)
    
    conflicts_ = get_conflicts(edge_allocation)
    print(colors.pure_red + f'Recomputed Coflicts')
    print(conflicts_)

end = time.time()
t_end_post = time.time()
print(colors.bright_white+f'Time Taken for Post Processing: {t_end_post-t_begin_post} seconds')
print(colors.bright_white+f'Total Time Taken: {end-start} seconds')
print('eprserved: ', sum(F.values()))
print(f'Total EPRs requested: {sum([D[i][2] for i in range(len(D))])}')

def debug():
    for f, v in F.items():
        if v>0:
            _, d_idx, p_idx, t = f.split('_')   # ['f', 'demand_idx', 'path_idx', 'time']
            # print(d_idx, p_idx, t)
            # print(P)
            # print(P[int(d_idx)][int(p_idx)])
            flow_str = f'At time: {t} for demand: {D[int(d_idx)]}'
            for e in P[int(d_idx)][int(p_idx)]:
                flow_str += f' -> (Edge {E[e]} with Capacity_{C[e]})'
            flow_str += f' : allocted {v} EPR'
            print(flow_str)
debug()



