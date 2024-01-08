import numpy as np
import networkx as nx
import math
import colors
import sys
import time
import random

v = 9
N, T = 20, 5
E = [(0, 1), (1, 2), (0, 4), (1, 3), (2, 5), (4, 3), (3, 5), (4, 6), (3, 7), (5, 8), (6, 7), (7, 8)]
C = [2, 3, 3, 2, 3, 4, 2, 2, 1, 3, 2, 5]  #(edge id, capacity)

scale, epsilon = 1, 0.1
U = 1

E_dict = {}
for idx, e in enumerate(E):
    E_dict[e] = idx

G = nx.Graph(E)
print(G)
nx.draw_networkx(G)
W_adj = nx.to_numpy_array(G)
print(W_adj)

def demands_to_algo(start_nodes, end_nodes, D, DST, DT):
    D_matrix = []
    for i in range(N):
        D_matrix.append((start_nodes[i], end_nodes[i], D[i], DST[i], DT[i]))
    return D_matrix

def youngs(v, N, T):
    ###In this N -> number of requests (k), len(E) -> number of edges (packing constraints)
    # Create the graph and demand set 
    #Test Case 1
    # v = 9
    # N, T = 20, 5
    # E = [(0, 1), (1, 2), (0, 4), (1, 3), (2, 5), (4, 3), (3, 5), (4, 6), (3, 7), (5, 8), (6, 7), (7, 8)]
    # C = [2, 3, 3, 2, 3, 4, 2, 2, 1, 3, 2, 5]  #(edge id, capacity)
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
        #print(start_times)
        #print(end_times)

        return start_nodes, end_nodes, demand_sizes, start_times, end_times
        

    start_nodes, end_nodes, D_size, DST, DT = generate_demands(nb_demands=N, nb_nodes=v, nb_timesteps=T, max_eprs=5)
    D = demands_to_algo(start_nodes, end_nodes, D_size, DST, DT)
    #print(D)

    # E_dict = {}
    # for idx, e in enumerate(E):
    #     E_dict[e] = idx

    # G = nx.Graph(E)
    # #print(G)
    # nx.draw_networkx(G)

    # Find the candidate paths for all demands
    P = {}
    edge_path_map = {}
    for i in range(N):
        all_p = []
        for p in nx.all_simple_paths(G, D[i][0], D[i][1], 3):
            #print(f'demand: {i} path: {p}')
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

    #print(P)

    # scale = np.ceil(np.log((len(C)*(T+1)+len(D))*(len(D)**(1+epsilon)/(1-(epsilon/2)))))
    # #print(scale)

    # scale = np.log(len(E)*T*N) 

    scale = 1
    #print(scale)

    start = time.time()
    # Set u_{i,p,t} values
    u = {}
    for i in range(N):
        for idx, p in enumerate(P[i]):
            for t in range(D[i][3], D[i][4]+1):
                u[f'u_{i}_{idx}_{t}'] = 1/(N*D[i][2])
                #print(u)

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
            # Y[f'y_{e}_{t}'] = np.exp((scale*H[f'h_{e}_{t}'])/(epsilon*C[e]))
             Y[f'y_{e}_{t}'] = 1

    #print(Y)
    #q_i
    Q = {}
    for i in range(N):
        # Q[f'q_{i}'] = np.exp((scale*L[f'l_{i}'])/(epsilon*D[i][2]))
        Q[f'q_{i}'] = 1
    #print(Q)

    #z_i
    Z = {}
    for i in range(N):
        # Z[f'z_{i}'] = np.exp(-(scale*L[f'l_{i}'])/(epsilon*alpha*D[i][2]))
        Z[f'z_{i}'] = 1
    #print(Z)

    r = np.exp(-(scale*g)/(epsilon*U))
    #print(r)

    """
        Calendering paper implementation
    """
    def should_loop():
        for i in range(N):
            #print(colors.yellow + f'inside should loop alpha[{i}], L[l_{i}], alpha[{i}]*D[i][2], U, g: {alpha[i], L[f"l_{i}"], alpha[i]*D[i][2], U, g}')
            # if L[f'l_{i}'] < alpha[i]*D[i][2] or g < U:
            # l_i = li(i)
            # #print(l_i)
            # if  l_i < alpha[i]*D[i][2]:
            if L[f'l_{i}'] < alpha[i]*D[i][2]:
                #print(colors.bright_white + f'yes loop, req_details: {D[i]}')
                return True
        #print(colors.bright_white + 'no - dont loop')
        return False

    def find_req():
        for i in range(N):
            for t in range(D[i][3], D[i][4]+1):
                for idx, p in enumerate(P[i]):
                    #print(colors.bright_white + f'{i, t, idx}')
                    #print(colors.pure_red+f'demand: {D[i]} and time: {t}')
                    #print(colors.dark_green +f'(edge index, edge, capacity) :{[(e, E[e], C[e]) for e in p]}')
                    #print(f'Edges: {E}')
                    #print(f'Capacity: {C}')
                    #print(f'Y: {Y}')
                    #print(f'Q: {Q}')
                    left_side_num = sum([Y[f'y_{e}_{t}']/C[e] for e in p])+Q[f'q_{i}']/D[i][2]
                    left_side_den = sum(Y.values()) + sum(Q.values())

                    # #print(f'left_side_den: {left_side_den}')
                    # left_side_den = 0
                    # for e in range(len(E)):
                    #     for t_prime in range(T+1):
                    #         left_side_den += Y[f'y_{e}_{t_prime}']
                    # for j in range(N):
                    #     left_side_den += Q[f'q_{j}']
                    # #print(f'left_side_den: {left_side_den}')
                    #print(colors.light_green)
                    #print(f'Req {i, (D[i])} Time: {t} and Path:{[E[e] for e in p]}')
                    #print(colors.bold + f'left num: {left_side_num} den:{left_side_den}')

                    right_side_num = 0
                    l_i = li(i)
                    #print(colors.yellow + f'Z: {Z}')
                    #print(colors.bright_white)
                    #print(f'l_i: {l_i} and alpha[i]*D[i][2]: {alpha[i]*D[i][2]}')
                    if L[f'l_{i}'] < alpha[i]*D[i][2]:
                    # if l_i < alpha[i]*D[i][2]:
                        right_side_num += Z[f'z_{i}']/(alpha[i]*D[i][2])
                    # if g < U:
                    #     right_side_num += u[f'u_{i}_{idx}_{t}']*r/U

                    right_side_den = 0
                    for j in range(N):
                        if L[f'l_{j}'] < alpha[j]*D[j][2]:
                            # #print(f'added req: {D[j]}')
                            right_side_den+= Z[f'z_{j}']
                        # else:
                            #print(f'skipped req: {D[j]}')
                    # if g < U:
                    #     right_side_den+= r
                    
                    #print(colors.bold + f'right num: {right_side_num} den:{right_side_den}')
                    #print(colors.bold + f'left val: {left_side_num/left_side_den} right val: {right_side_num/right_side_den}')
                    if left_side_num/left_side_den <= right_side_num/right_side_den:
                        # #print(f'{i, t, idx}')
                        #print(colors.bright_white + 'found')
                        return (i, t, idx)
        #print(colors.bright_white)
        return None

    iter = 0
    # max_D = max([D[i][2] for i in range(len(D))])
    for a in np.arange(0.1, 1.1, 0.1):
        alpha = [a for i in range(len(D))]
    # for nEPR in range(1, max_D+1, 1):
        # alpha = [nEPR/D[i][2] if nEPR<D[i][2] else 1 for i in range(len(D))]
        #print(f'computed alpha: {alpha}')
        isInfeasible = False
        while should_loop():
            iter += 1
            #print(f'alpha: ------ {alpha} and iteration : {iter}')
            request = find_req()
            # break

            if request == None:
                #print(f'could not allocate at iteration: {iter}')
                break

            i_star, t_star, p_star = request
            #print(i_star, t_star, p_star)
            min_ = math.inf  #Finding min edge capacity along the path p_star
            for e in P[i_star][p_star]:
                # #print(P[i_star][p_star])
                e_capacity = C[e]
                if e_capacity < min_:
                    min_ = e_capacity

            gamma = epsilon*min(D[i_star][2], min_)
            #print(f'gamma: {gamma}')

            if L[f'l_{i_star}'] < alpha[i_star]*D[i_star][2]:
                gamma = min(gamma, epsilon*alpha[i_star]*D[i_star][2])
            
            #print(f'gamma: {gamma}')

            # if g < U:
            #     gamma = min(gamma, epsilon*U/u[f'u_{i_star}_{p_star}_{t_star}'])
            
            #print(f'gamma: {gamma}')
            
            for e in P[i_star][p_star]:
                #print(f"Bef updating: Y[f'y_{e}_{t_star}'] : {Y[f'y_{e}_{t_star}']}")
                Y[f'y_{e}_{t_star}'] = Y[f'y_{e}_{t_star}']*np.exp(gamma/C[e])
                #print(f"Aft updating: Y[f'y_{e}_{t_star}'] : {Y[f'y_{e}_{t_star}']}")     
            
            
            Q[f'q_{i_star}'] = Q[f'q_{i_star}']*np.exp(gamma/D[i_star][2])

            Z[f'z_{i_star}'] = Z[f'z_{i_star}']*np.exp(-gamma/(alpha[i_star]*D[i_star][2]))

            r = r*np.exp((-u[f'u_{i_star}_{p_star}_{t_star}']*gamma)/U)

            F[f'f_{i_star}_{p_star}_{t_star}'] = F[f'f_{i_star}_{p_star}_{t_star}'] + (epsilon/scale)*gamma
            #print(f'f_{i_star}_{p_star}_{t_star} updated by  {(epsilon/scale)*gamma}')
        
            #End of While
            compute_li()
            g = compute_g(g)
            #print(F)
            #print(g)
        
        # break

        if isInfeasible:
            #print(f'max alpha: {[a - 0.1 for a in alpha]}')
            break
    """
        End Calendering Paper implementation
    """


    """
        Begin Post Processing and Edge capacity checks
    """
    t_begin_post = time.time()
    #We have a prelimnary value of f now:
    #print(colors.bright_blue+'Computed f by Youngs method')
    #print(F)

    for f, v in F.items():
        F[f] = round(v)

    #print(colors.light_green + 'Clipped F by rounding')
    #print(F)   #Clipped values of f

    #Heursitic to correct the edge usage and adjust allocations
    alpha = [li(i)/D[i][2] for i in range(N)]
    #print(alpha)

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
    #print(colors.bright_white + 'The edge allocations are given below')
    #print(edge_allocation)

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
    #print(colors.pure_red+'detected conflicts')
    #print(conflicts_)

    def remove_flow(conflicts):
        #Sort the flows along each conflicted edge in descending order of alpha(fraction completed)
        for e, paths in conflicts.items():
            paths.sort(key=lambda tup: tup[2])

        #print(colors.light_green+'Sorted Conflicts')
        #print(conflicts)
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
        #print(colors.yellow + f'After removing flow with max alpha \nUpdated F:')
        #print(F)

        #print(colors.dark_green + f'Conflicts after removing flow')
        #print(conflicts_)
        
        edge_allocation = get_edge_allocations()
        #print(colors.bright_white + f'Recomputed Edge Allocations')
        #print(edge_allocation)
        
        conflicts_ = get_conflicts(edge_allocation)
        #print(colors.pure_red + f'Recomputed Coflicts')
        #print(conflicts_)

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

    D_alloc = [0 for i in range(len(D))]
    for i in range(len(D)):
        for p_idx, p in enumerate(P[i]):
            for t in range(D[i][3], D[i][4]+1):
                D_alloc[i] += F[f'f_{i}_{p_idx}_{t}']
    
    Demands = [D[i][2] for i in range(N)]
    print(f'Demands: {Demands}')
    print(f'Allocated: {D_alloc}')

    return start_nodes, end_nodes, D_size, DST, DT, Demands, D_alloc, end-start

"""
----------------------------
LP Part
----------------------------
"""
def lp(start_nodes, end_nodes, D, DST, DT, k, T):

    print('LP to find the optimal')
    import pandas as pd
    import numpy as np
    import random
    import networkx as nx
    import matplotlib.pyplot as plt
    from docplex.mp.model import Model
    from docplex.mp.vartype import VarType
    import json

    def add_variables(model, W_adj, nb_demands, nb_timesteps, nb_nodes, debug=False):
        variables = model.integer_var_dict(
            [
                (i, t, u, v)
                for i in range(nb_demands)
                for t in range(nb_timesteps)
                for u in range(nb_nodes)
                for v in range(nb_nodes)
                if W_adj[u][v] != 0
            ],
            name="f",
        )
        if debug:
            for var_key in variables:
                print(variables[var_key].name)
        return variables

    def define_objective(model, W_adj, nb_demands, nb_nodes, demand_list, variables, debug=False):
        objective = 0
        for i in range(nb_demands):
            for t in range(demand_list[i][3], demand_list[i][4], 1):
                for v in range(nb_nodes):
                    if W_adj[demand_list[i][0]][v] != 0:
                        if debug: 
                            print(
                                (i, t, demand_list[i][0], v),
                                variables[(i, t, demand_list[i][0], v)],
                            )
                            print()
                        objective += (
                            variables[(i, t, demand_list[i][0], v)]
                        )
        model.maximize(objective)

    def define_constraints(
        model,
        W_adj,
        nb_timesteps,
        nb_demands,
        nb_nodes,
        demand_list,
        variables,
        nw_state,
        debug=False,
    ):
        # Flow Conservation Constraint:
        for i in range(nb_demands):
            for t in range(nb_timesteps):
                for v in range(nb_nodes):
                    if v not in (demand_list[i][0], demand_list[i][1]):
                        inflow_v = model.linear_expr()
                        outflow_v = model.linear_expr()
                        for u in range(nb_nodes):
                            if W_adj[u][v] != 0:
                                inflow_v += variables[(i, t, u, v)]
                                outflow_v += variables[(i, t, v, u)]
                        if debug:
                            print(f"INFLOW_I_{i}_T_{t}_V_{v} = {inflow_v}")
                            print(f"OUTFLOW_I_{i}_T_{t}_V_{v} = {outflow_v}")
                        model.add_constraint(
                            inflow_v == outflow_v, f"flow_cons_node_i_{i}_t_{t}_v_{v}"
                        )
                        if debug:
                            print(
                                model.get_constraint_by_name(
                                    f"flow_cons_node_i_{i}_t_{t}_v_{v}"
                                )
                            )

        # The Capacity Constraint:
        for t in range(nb_timesteps):
            for u in range(nb_nodes):
                for v in range(nb_nodes):
                    if W_adj[u][v] != 0:
                        sum_allocated = model.linear_expr()
                        for i in range(nb_demands):
                            sum_allocated += variables[(i, t, u, v)] + variables[(i, t, v, u)]
                        model.add_constraint(
                            sum_allocated <= nw_state[t][u][v],
                            f"edge_capacity_{u}_{v}_t_{t}",
                        )
                        if debug:
                            print(
                                model.get_constraint_by_name(f"edge_capacity_{u}_{v}_t_{t}")
                            )
                            print()
        
        # The Out-of-Window Constraint:
        total_allocation = model.linear_expr()
        for i in range(nb_demands):
            demand_allocation = model.linear_expr()
            dont_allocate_time = [
                t
                for t in range(0, nb_timesteps)
                if t not in range(demand_list[i][3], demand_list[i][4])
            ]
            if debug:
                print(
                    f"dont allocate time for req: {demand_list[i]} = {dont_allocate_time}"
                )
            for t in dont_allocate_time:
                for u in range(nb_nodes):
                    for v in range(nb_nodes):
                        if W_adj[u][v] != 0:  # edge u->v
                            demand_allocation += variables[(i, t, u, v)]
            total_allocation += demand_allocation 
            # model.add_constraint(sum_allocated == 0, f"no_allocation_bef_after_req_{i}")
        # DEBUG PRINTING
        model.add_constraint(total_allocation == 0, f"no_oow_allocation") 
        if debug:
            print(model.get_constraint_by_name(f"no_oow_allocation"))
            print()
        
        # The Demand Size Constraint:
        for i in range(nb_demands):
            source_outflow = model.linear_expr()
            for t in range(demand_list[i][3], demand_list[i][4], 1):
                sum_allocated = model.linear_expr()
                u = demand_list[i][0]
                for v in range(nb_nodes):
                    if W_adj[u][v] != 0:  # edge from  u -> v
                        source_outflow += variables[(i, t, u, v)]
                # model.add_constraint(
                #     sum_allocated <= demand_list[i][2],
                #     f"max_allocation_demand_{i}_t_{t}",
                # )

            dest_inflow = model.linear_expr()
            for t in range(demand_list[i][3], demand_list[i][4], 1):
                sum_allocated = model.linear_expr()
                v = demand_list[i][1]
                for u in range(nb_nodes):
                    if W_adj[u][v] != 0:  # edge from  u -> v
                        dest_inflow += variables[(i, t, u, v)]
                # model.add_constraint(
                #     sum_allocated <= demand_list[i][2],
                #     f"max_allocation_demand_{i}_t_{t}",
                # )
            model.add_constraint(
                source_outflow <= demand_list[i][2],
                f"demand_size_{i}"
            )
            model.add_constraint(
                dest_inflow == source_outflow,
                f"dest_demand_size_{i}"
            )
            if debug:
                print(model.get_constraint_by_name(f"max_allocation_demand_{i}_t_{t}"))
                print()

        #Any inflow into the source vertex should be 0 wrt that demand and outflow at the dest vertex should be 0
        source_inflow_dest_outflow = model.linear_expr()
        for i in range(nb_demands):
            #incoming edges into src
            for t in range(demand_list[i][3], demand_list[i][4], 1):
                for v in range(nb_nodes):
                    if W_adj[v][demand_list[i][0]] != 0:
                        source_inflow_dest_outflow += variables[(i, t, v, demand_list[i][0])]
            
            #outgoing edges from dest
            for t in range(demand_list[i][3], demand_list[i][4], 1):
                for v in range(nb_nodes):
                    if W_adj[demand_list[i][1]][v] != 0:
                        source_inflow_dest_outflow += variables[(i, t, demand_list[i][1], v)]
        
        model.add_constraint(
                source_inflow_dest_outflow == 0,
                "inflow_src_outflow_dst"
            )
        
        print(model.get_constraint_by_name("inflow_src_outflow_dst"))
        print()

    def generate_flow_matrix(vars, nb_demands, nb_timesteps, nb_nodes, debug=False):
        F = np.zeros(shape=(nb_demands, nb_timesteps, nb_nodes, nb_nodes))
        # print("")
        # print("")
        # print("FLOWS")
        # print("-------------------")
        # print("[i, t, u, v] | flow")
        # print("-------------------")
        for v in vars:
            if v["value"] != 0:
                id = v["name"]
                id = id.split("_")[1:]
                id = list(map(int, id))
                F[id[0], id[1], id[2], id[3]] = float(v["value"])
                # print(f"{id} | {v['value']}")
        # print("-------------------")
        return F

    def compute_served(F, demand_list):
        nb_demands = len(demand_list)
        served_count = 0
        total_eprs = 0
        alloc = [0 for i in range(nb_demands)]
        for i in range(nb_demands):
            eprs_served = 0
            for t in range(F.shape[1]):
                for v in range(F.shape[2]):
                    eprs_served += F[i][t][demand_list[i][0]][v]
            if eprs_served == demand_list[i][2]:
                served_count += 1
                # print(f"Served demand {i}")
            alloc[i] = eprs_served
            total_eprs += eprs_served

        # print(f"Served {served_count} out of {nb_demands} demands.")
        return served_count, total_eprs, alloc
        

    from networkx.generators.random_graphs import erdos_renyi_graph
    import networkx as nx
    from numpy import arange
    import time
    import sys


    # Need to increase T by 1 and the demand deadlines by 1
    T += 1
    DT = list(DT)
    for i in range(k):
        DT[i] += 1

    #LP Part
    network_state = [W_adj] * (T + 1)
    Tstart_lp = time.perf_counter()
    model = Model(name="routing", log_output=False)
    model.objective_sense = "max"
    model.parameters.threads.set(1)
    variables = add_variables(model=model, W_adj=W_adj,nb_demands=k, nb_timesteps=T, nb_nodes=v)
    # define_multiobjective(model=model, W_adj=W_adj, nb_demands=k, nb_nodes=v, demand_list=list(zip(start_nodes, end_nodes, D, DST, DT)), variables=variables)                                                                     
    define_objective(model=model, W_adj=W_adj, nb_demands=k, nb_nodes=v, demand_list=list(zip(start_nodes, end_nodes, D, DST, DT)), variables=variables)
    define_constraints(model=model, W_adj=W_adj, nb_timesteps=T, nb_demands=k, nb_nodes=v, demand_list=list(zip(start_nodes, end_nodes, D, DST, DT)), variables=variables, nw_state=network_state)
    solution = model.solve()
    Tend_lp = time.perf_counter()
    print(f'Time taken LP: {Tend_lp-Tstart_lp}')

    if solution is None:
        dserved_lp = eprsserved_lp = 0
    else:
        sol_json = solution.export_as_json_string()
        vars = json.loads(sol_json)["CPLEXSolution"]["variables"]
        F = generate_flow_matrix(vars, nb_demands=k, nb_timesteps=10, nb_nodes=v)
        dserved_lp, eprserved_lp, alloc = compute_served(F, list(zip(start_nodes, end_nodes, D, DST, DT)))
        print('eprserved_lp: ', eprserved_lp)
    
    print(f'total EPR requested: {sum(D)}')
    return alloc, Tend_lp-Tstart_lp

def youngs_online(start_nodes, end_nodes, D_size, DST, DT, N, T):
    
    start = time.time()
    
    D = demands_to_algo(start_nodes, end_nodes, D_size, DST, DT)
    print(D)

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

    #L_i
    L = {}
    for i in range(N):
        for tau in range(T+1):
            L[f'l_{i}_{tau}'] = 0

    #Demand Satisfied  --- (2)
    def l_i_tau(i, tau):
        L[f'l_{i}_{tau}'] = 0
        for t in range(max(tau, D[i][3]), D[i][4]+1):
            for idx, p in enumerate(P[i]):
                L[f'l_{i}_{tau}'] += F[f'f_{i}_{idx}_{t}']
        return L[f'l_{i}_{tau}']

    def F_i_tau(i, tau):
        total_alloc = 0
        for t in range(D[i][3], tau+1):
            for idx, p in enumerate(P[i]):
                total_alloc += F[f'f_{i}_{idx}_{t}']
        return total_alloc


    """
    Variables initialization
    """
    #y_{e,t}
    Y = {}
    for e in range(len(E)):
        for t in range(T+1):
            # Y[f'y_{e}_{t}'] = np.exp((scale*H[f'h_{e}_{t}'])/(epsilon*C[e]))
            Y[f'y_{e}_{t}'] = 1
    print(Y)

    #q_i
    Q = {}
    for i in range(N):
        # Q[f'q_{i}'] = np.exp((scale*L[f'l_{i}'])/(epsilon*D[i][2]))
        Q[f'q_{i}'] = 1
    print(Q)

    #z_i
    Z = {}
    for i in range(N):
        # Z[f'z_{i}'] = np.exp(-(scale*L[f'l_{i}'])/(epsilon*alpha*D[i][2]))
        Z[f'z_{i}'] = 1
    print(Z)


    def R(tau): #relevant demands
        valid_d = []
        for i in range(N):
            if tau>=D[i][3] and tau<=D[i][4]:
                valid_d.append(i)
        return valid_d

    Beta, C_e_t, c = {}, {}, 10
    for e in range(len(E)):
        for tau in range(1, T):
            for t in range(tau, T+1):
                Beta[f'b_{e}_{t}_{tau}'] = np.exp(-(t-tau)/c)
                C_e_t[f'C_{e}_{t}_{tau}'] = math.floor(Beta[f'b_{e}_{t}_{tau}']*C[e])


    """
        Calendering paper implementation
    """
    def should_loop(tau):
        for i in R(tau):
            print(colors.yellow + f'inside should loop alpha[{i}], L[l_{i}_{tau}], alpha[{i}]*D[i][2]: {alpha[i], L[f"l_{i}_{tau}"], alpha[i]*D[i][2]}')
            if L[f'l_{i}_{tau}'] < alpha[i]*D[i][2] - F_i_tau(i, tau-1):
                print(colors.bright_white + f'yes loop, req_details: {D[i]}')
                return True
        print(colors.bright_white + 'no - dont loop')
        return False

    def find_req(tau):
        for i in R(tau):
            for t in range(max(tau, D[i][3]), D[i][4]+1): #The flow variable should be >= the current timestep tau
                for idx, p in enumerate(P[i]):
                    print(colors.bright_white + f'{i, t, idx}')
                    print(colors.pure_red+f'demand: {D[i]} and time: {t}')
                    print(colors.dark_green +f'(edge index, edge, capacity) :{[(e, E[e], C[e]) for e in p]}')
                    print(f'Edges: {E}')

                    flag_path = False
                    for e in p:
                        print('Capcity', C[e], 'Allocable Capacity: ', C_e_t[f'C_{e}_{t}_{tau}'])
                        #if any edges allocable capcity is 0, skip the path
                        if C_e_t[f'C_{e}_{t}_{tau}'] == 0:
                            flag_path = True
                            break
                    if flag_path:
                        continue
                    
                    print(f'Y: {Y}')
                    print(f'Q: {Q}')
                    left_side_num = sum([Y[f'y_{e}_{t}']/C[e] for e in p])+Q[f'q_{i}']/(D[i][2]-F_i_tau(i, tau-1))
                    left_side_den = 0
                    for e in range(len(E)):
                        for t_prime in range(tau, T+1):
                            left_side_den += Y[f'y_{e}_{t_prime}']
                    
                    for i_prime in R(tau):
                        left_side_den += Q[f'q_{i_prime}']


                    print(colors.light_green)
                    print(f'Req {i, (D[i])} Time: {t} and Path:{[E[e] for e in p]}')
                    print(colors.bold + f'left num: {left_side_num} den:{left_side_den}')

                    right_side_num = 0
                    l_i_tao = l_i_tau(i, tau)
                    print(colors.yellow + f'Z: {Z}')
                    print(colors.bright_white)
                    print(f'l_i: {l_i_tao} and alpha[i]*D[i][2]: {alpha[i]*D[i][2]}')
                    if l_i_tao < alpha[i]*D[i][2] - F_i_tau(i, tau-1):
                        right_side_num += Z[f'z_{i}']/(alpha[i]*D[i][2] - F_i_tau(i, tau-1))

                    right_side_den = 0
                    for j in R(tau):
                        l_j_tao = l_i_tau(j, tau)
                        print(f'Req: {j} l_j_tao: {l_j_tao} and alpha[j]*D[j][2]: {alpha[j]*D[j][2]}')
                        if l_j_tao < alpha[j]*D[j][2] - F_i_tau(j, tau-1):
                            right_side_den+= Z[f'z_{j}']
                    
                    if right_side_num == 0: #To handle division by 0 error
                        right_side_den = 1

                    print(colors.bold + f'right num: {right_side_num} den:{right_side_den}')
                    print(colors.bold + f'left val: {left_side_num/left_side_den} right val: {right_side_num/right_side_den}')
                    if left_side_num/left_side_den <= right_side_num/right_side_den:
                        # print(f'{i, t, idx}')
                        print(colors.bright_white + 'found')
                        return (i, t, idx)
        print(colors.bright_white)
        return None

    iter = 0

    for tau in range(1, 2):
        print(f'T: {tau}')
        t_start_tau = time.time()
        # max_D = max([D[i][2] for i in range(len(D))])
        for a in np.arange(0.1, 1.1, 0.1):
            alpha = [a for i in range(len(D))]
        # for nEPR in range(1, max_D+1, 1):
            # alpha = [nEPR/D[i][2] if nEPR<D[i][2] else 1 for i in range(len(D))]
            print(f'computed alpha: {alpha}')
            isInfeasible = False
            while should_loop(tau):
                iter += 1
                print(f'alpha: ------ {alpha} and iteration : {iter}')
                request = find_req(tau)
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

                if L[f'l_{i_star}_{tau}'] < alpha[i_star]*D[i_star][2]-F_i_tau(i_star, tau-1):
                    gamma = min(gamma, epsilon*(alpha[i_star]*D[i_star][2]-F_i_tau(i_star, tau-1)))
                
                print(f'gamma: {gamma}')

                # if g < U:
                #     gamma = min(gamma, epsilon*U/u[f'u_{i_star}_{p_star}_{t_star}'])
                
                print(f'gamma: {gamma}')
                
                for e in P[i_star][p_star]:
                    print(f"Bef updating: Y[f'y_{e}_{t_star}'] : {Y[f'y_{e}_{t_star}']}")
                    Y[f'y_{e}_{t_star}'] = Y[f'y_{e}_{t_star}']*np.exp(gamma/C_e_t[f'C_{e}_{t_star}_{tau}'])
                    print(f"Aft updating: Y[f'y_{e}_{t_star}'] : {Y[f'y_{e}_{t_star}']}")     
                
                
                Q[f'q_{i_star}'] = Q[f'q_{i_star}']*np.exp(gamma/(D[i_star][2]-F_i_tau(i_star, tau-1)))

                Z[f'z_{i_star}'] = Z[f'z_{i_star}']*np.exp(-gamma/(alpha[i_star]*D[i_star][2]-F_i_tau(i_star, tau-1)))

                F[f'f_{i_star}_{p_star}_{t_star}'] = F[f'f_{i_star}_{p_star}_{t_star}'] + (epsilon/scale)*gamma
                print(f'f_{i_star}_{p_star}_{t_star} updated by  {(epsilon/scale)*gamma}')
            
                print(F)
            
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
        alpha = [l_i_tau(i, 0)/D[i][2] for i in range(N)]
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

        t_end_tau = time.time()
        t_end_post = time.time()
        print(colors.bright_white+f'Time Taken for Post Processing: {t_end_post-t_begin_post} seconds')
        print(colors.bright_white+f'Total Time Taken for tau : {t_end_tau-t_start_tau} seconds')

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
    
    end = time.time()
    print(f'time taken: {end-start}')
    print('eprserved: ', sum(F.values()))
    print(f'Total EPRs requested: {sum([D[i][2] for i in range(len(D))])}')

    D_alloc = [0 for i in range(len(D))]
    for i in range(len(D)):
        for p_idx, p in enumerate(P[i]):
            for t in range(D[i][3], D[i][4]+1):
                D_alloc[i] += F[f'f_{i}_{p_idx}_{t}']
    
    return D_alloc, end-start




start_nodes, end_nodes, D_size, DST, DT, Demands, D_alloc, t_exec_yng = youngs(v, N, T)
lp_alloc, t_exec_lp = lp(start_nodes, end_nodes, D_size, DST, DT, N, T)
yo_alloc, t_exec_yo = youngs_online(start_nodes, end_nodes, D_size, DST, DT, N, T)
lp_alloc = [int(i) for i in lp_alloc]
print(f'D: {Demands}')
print(f'Y: {D_alloc}')
print(f'L: {lp_alloc}')
print(f'YO: {yo_alloc}')

print(f'Total EPR Req: {sum(Demands)}')
print(f'eprserved_yng: {sum(D_alloc)}')
print(f'eprserved_lp: {sum(lp_alloc)}')
print(f'eprserved_yo: {sum(yo_alloc)}')

print(f'Time Taken Yng: {t_exec_yng}')
print(f'Time Taken LP: {t_exec_lp}')
print(f'Time Taken YO: {t_exec_yo}')
