import numpy as np
import networkx as nx
import math
import colors

# scale, epsilon = 1, 0.1
scale, epsilon = 1, 0.01
alpha, U = 1, 1   #U=0, doesn't work -> division by 0 in r


# Create the graph
E = [(1, 2), (2, 3), (2, 4)]
C = [2, 1, 1]  #(edge id, capacity)
E_dict = {}
for idx, e in enumerate(E):
    E_dict[e] = idx

G = nx.Graph(E)
print(G)
nx.draw_networkx(G)

# Create demand set
N, T = 2, 3

D = [(1, 3, 5, 1, 3), (1, 4, 3, 1, 2)] #(s, e, d, ST, ET)
# D = [(1, 3, 2, 1, 3), (1, 4, 1, 1, 1)]

# Find the candidate paths for all demands
P = {}
for i in range(N):
    all_p = []
    for p in nx.all_simple_paths(G, D[i][0], D[i][1], 2):
        p_idx = []
        for j in range(len(p)-1):
            p_idx.append(E_dict[(p[j], p[j+1])])
        all_p.append(p_idx)
    P[i] = all_p

print(P)

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

def should_loop():
    for i in range(N):
        print(colors.yellow + f'inside should loop alpha, L[l_{i}], alpha*D[i][2], U, g: {alpha, L[f"l_{i}"], alpha*D[i][2], U, g}')
        # if L[f'l_{i}'] < alpha[i]*D[i][2] or g < U:
        # l_i = li(i)
        # print(l_i)
        # if  l_i < alpha[i]*D[i][2]:
        if L[f'l_{i}'] < alpha[i]*D[i][2]:
            print(colors.bright_white + 'yes - loop')
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
                
                right_side_num = 0
                l_i = li(i)
                print(f'l_i: {l_i} and alpha[i]*D[i][2]: {alpha[i]*D[i][2]}')
                print(f'Z: {Z}')
                if L[f'l_{i}'] < alpha[i]*D[i][2]:
                # if l_i < alpha[i]*D[i][2]:
                    right_side_num += Z[f'z_{i}']/(alpha[i]*D[i][2])
                # if g < U:
                #     right_side_num += u[f'u_{i}_{idx}_{t}']*r/U

                right_side_den = 0
                for j in range(N):
                    if L[f'l_{j}'] < alpha[j]*D[j][2]:
                        right_side_den+= Z[f'z_{j}']
                # if g < U:
                #     right_side_den+= r
                
                # print(f'{i, t, idx}')
                print(colors.bold + f'left num: {left_side_num} den:{left_side_den}')
                print(colors.bold + f'right num: {right_side_num} den:{right_side_den}')
                print(colors.bold + f'left val: {left_side_num/left_side_den} right val: {right_side_num/right_side_den}')
                if left_side_num/left_side_den <= right_side_num/right_side_den:
                    # print(f'{i, t, idx}')
                    print(colors.bright_white + 'found')
                    return (i, t, idx)
    print(colors.bright_white)
    return None

# for alpha in np.arange(0.1, 1.1, 0.1):
max_D = max([D[i][2] for i in range(len(D))])
for nEPR in range(1, max_D+1, 1):
    alpha = [nEPR/D[i][2] if nEPR<D[i][2] else 1 for i in range(len(D))]
    print(f'computed alpha: {alpha}')
    isInfeasible = False
    while should_loop():
        print(f'alpha: ------ {alpha}')
        request = find_req()
        # break

        if request == None:
            isInfeasible = True
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