import pandas as pd
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from docplex.mp.model import Model
from docplex.mp.vartype import VarType
import json

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

    return start_nodes, end_nodes, demand_sizes, start_times, end_times

def plot_digraph(G, filename):
    plt.figure()
    positions = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, positions)
    nx.draw_networkx_edges(G, pos=positions, width=1.0)
    edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, positions, edge_labels=edge_labels)
    nx.draw_networkx_labels(G, positions)
    # plt.savefig(filename)
    plt.show()

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
    for i in range(nb_demands):
        eprs_served = 0
        for t in range(F.shape[1]):
            for v in range(F.shape[2]):
                eprs_served += F[i][t][demand_list[i][0]][v]
        if eprs_served == demand_list[i][2]:
            served_count += 1
            # print(f"Served demand {i}")
        total_eprs += eprs_served

    # print(f"Served {served_count} out of {nb_demands} demands.")
    return served_count, total_eprs
    

from networkx.generators.random_graphs import erdos_renyi_graph
import networkx as nx
from numpy import arange
import time
import sys

v = 9 #Test Case 1
k, T = 20, 5
E = [(0, 1), (1, 2), (0, 4), (1, 3), (2, 5), (4, 3), (3, 5), (4, 6), (3, 7), (5, 8), (6, 7), (7, 8)]
C = [2, 3, 3, 2, 3, 4, 2, 2, 1, 3, 2, 5]  #(edge id, capacity)


# v = 4   #Test Case 2
# k, T = 2, 3
# E = [(0, 1), (1, 2), (1, 3), (2, 3)]
# C = [2, 2, 2, 2] 


G = nx.Graph()
G.add_weighted_edges_from([(E[i][0], E[i][1], C[i]) for i in range(len(E))])
nx.draw_networkx(G)
W_adj = nx.to_numpy_array(G)
print(W_adj)


# Need to increase T by 1 and the demand deadlines by 1
T += 1
def demands_from_algo():
    #Test Case 1
    # D_matrix = [(0, 2, 2, 1, 3),
    #             (0, 3, 2, 2, 3),
    #             (0, 5, 3, 1, 4),
    #             (2, 7, 2, 3, 4),
    #             (3, 6, 3, 4, 5)]
    D_matrix = [(0, 2, 2, 1, 3),
        (0, 3, 5, 2, 3),
        (0, 5, 2, 1, 4),
        (2, 7, 5, 3, 4),
        (3, 6, 4, 4, 5),
        (1, 3, 4, 1, 2),
        (4, 7, 2, 2, 5),
        (2, 7, 3, 3, 4)]
    #Test Case 2  
    # D_matrix = [(0, 3, 4, 1, 3), (0, 2, 3, 1, 3)]  

    #Formatting Demands as understood by LP
    start_nodes = []
    end_nodes = []
    DST = []
    DT = []
    D = []

    for i in range(k):
        start_nodes.append(D_matrix[i][0])
        end_nodes.append(D_matrix[i][1])
        D.append(D_matrix[i][2])
        DST.append(D_matrix[i][3])
        DT.append(D_matrix[i][4]+1)
    
    return start_nodes, end_nodes, D, DST, DT

def demands_to_algo(start_nodes, end_nodes, D, DST, DT):
    D_matrix = []
    for i in range(k):
        D_matrix.append((start_nodes[i], end_nodes[i], D[i], DST[i], DT[i]))
    print(D_matrix)

# start_nodes, end_nodes, D, DST, DT = demands_from_algo()
start_nodes, end_nodes, D, DST, DT = generate_demands(nb_demands=k, nb_nodes=v, nb_timesteps=T, max_eprs=5)
demands_to_algo(start_nodes, end_nodes, D, DST, DT)

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
    dserved_lp, eprserved_lp = compute_served(F, list(zip(start_nodes, end_nodes, D, DST, DT)))
    print('eprserved_lp: ', eprserved_lp)