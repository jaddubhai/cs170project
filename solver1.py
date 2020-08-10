import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import sys
import numpy as np


def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """
    T = MRCT(G)
    return T

def MRCT(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph - a Minimum Routing Cost Tree for G
    """
    L, NodeLevel, l_max, n = network_reduction(G)
    T, L, Nodevel, l_max = build_tree(G, L, NodeLevel, l_max, n)
    #T = optimize(G, T, L, NodeLevel, l_max)
    return T
    

def network_reduction(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        L: dictionary mapping levels to a list of the nodes in them
        NodeLevel: dictionary mapping nodes in G to their level
        l_max: the level of the core node(s)
        n: dictionary mapping edges to the number of pairs (u, v) for which the edge is used
        in the shortest path between u and v in some shortest path tree T, summed over all T
    """
    V = list(G.nodes)
    n = {}
    n_ij = {}
    for k in V:
        lengths, T_k = nx.single_source_dijkstra(G, k, weight='weight')
        edges = set()
        for path in T_k.values():
            [edges.add(tuple(sorted([path[i - 1], path[i]]))) for i in range(1, len(path))]
        T_k = nx.Graph()
        T_k.add_edges_from(list(edges))
        assert len(T_k.edges()) == len(V) - 1
        n_ij[k] = get_n_ij(edges, T_k)
    E = G.edges()
    for e in E: 
        n[e] = sum([n_ij[k][e] for k in V if e in n_ij[k].keys()])
    keys = list(n.keys())
    for key in keys:
        n[(key[1], key[0])] = n[key]
    V_new = V
    L = {}
    NodeLevel = {}
    l = 0
    while len(V_new) > 0:
        l += 1
        deg = {}
        for k in V_new:
            deg[k] = sum([n[(k, j)] for j in V_new if (k, j) in n.keys()])
        deg_min = min([deg[k] for k in V_new])
        L[l] = [k for k in V_new if deg[k] == deg_min]
        for k in L[l]:
            V_new.remove(k)
        for k in L[l]:
            NodeLevel[k] = l
    l_max = l
    return (L, NodeLevel, l_max, n)

def get_n_ij(edges, T):
    """
    Args: 
        edges: list of edges (u, v)
        T: networkx.Graph - shortest path tree

    Returns:
        n: dictionary mapping edges to the number of pairs (u, v) for which the edge is used
        in the shortest path between u and v in T
    """
    n = {}
    for edge in edges:
        n[edge] = 0
    V = list(T.nodes)
    for u in V:
        for v in V:
            if u == v:
                pass
            else:
                path = list(nx.all_simple_paths(T, u, v))
                assert len(path) == 1
                path = path[0]
                for i in range(1, len(path)):
                    v1 = path[i - 1]
                    v2 = path[i]
                    n[tuple(sorted([v1, v2]))] += 1
    keys = list(n.keys())
    for key in keys:
        n[(key[1], key[0])] = n[key]
    return n

def build_tree(G, L, NodeLevel, l_max, n):
    """
    Args:
        G: networkx.Graph
        L: dictionary mapping levels to a list of the nodes in them
        NodeLevel: dictionary mapping nodes in G to their level
        l_max: the level of the core node(s)
        n: dictionary mapping edges to the number of pairs (u, v) for which the edge is used
        in the shortest path between u and v in some shortest path tree T, summed over all T

    Returns:
        MRCT: networkx.Graph - spanning tree of G created by greedily adding edges with maximal n
        values to the core node
    """
    T = nx.Graph()
    deg = {}
    flag = 0
    for k in G.nodes:
        deg[k] = sum([n.get((k, j), 0) for j in G.nodes])
    core = [k for k in G.nodes if NodeLevel[k] == l_max]
    if len(core) == 1:
        T.add_node(core[0])
    else:
        T.add_node(max(core, key = lambda x : deg[x]))
    l = l_max
    unconnected = []
    for l in range(l_max, 0, -1):
        j = 0
        while j < len(unconnected):
            k = unconnected[j]
            #i = max(T, key = lambda x : weight_heuristic(x, k, n, G))
            V = list(T.nodes)
            i = min(V, key = lambda x : test_edge(T, G, x, k))
            if (i, k) not in G.edges:
                pass
            else:
                T.add_node(k)
                T.add_weighted_edges_from([(i, j, G[i][k]['weight'])])
                unconnected.remove(k)
            j += 1
        for k in L[l]:
            if k in T:
                pass
            else:
                #i = max(T, key = lambda x : weight_heuristic(x, k, n, G))
                V = list(T.nodes)
                i = min(V, key = lambda x : test_edge(T, G, x, k))
                if (i, k) not in G.edges:
                    unconnected.append(k)
                else:
                    T.add_node(k)
                    T.add_weighted_edges_from([(i, k, G[i][k]['weight'])])
                    if nx.is_dominating_set(G, T):
                        flag = 1
                        break
        if flag == 1:
            break
    return T, L, NodeLevel, l_max

def test_edge(T, G, u, v):
    if (u, v) not in G.edges:
        return float("inf")
    T.add_weighted_edges_from([(u, v, G[u][v]['weight'])])
    d = average_pairwise_distance(T)
    T.remove_edge(u, v)
    return d

def weight_heuristic(u, v, n, G):
    """
    Args:
        G: networkx.Graph
        u: node in G
        v: node in G
        n: dictionary mapping edges to the number of pairs (u, v) for which the edge is used
        in the shortest path between u and v in some shortest path tree T, summed over all T

    Returns:
        A heuristic representing how likely the edge (u, v) is to be in the MRCT solution
    """
    if (u, v) not in G.edges:
        return 0
    return n[(u, v)] / G[u][v]['weight']

def optimize(G, T, L, NodeLevel, l_max):
    """
    Args:
        G: networkx.Graph
        T: networkx.Graph - MRCT for G
        L: dictionary mapping levels to a list of the nodes in them
        NodeLevel: dictionary mapping nodes in G to their level
        l_max: the level of the core node(s)

    Returns:
        T: networkx.Graph - modified version of input T optimized for reduced average
        pairwise distance
    """
    core = [k for k in T.nodes if NodeLevel[k] == l_max]
    min_cost = average_pairwise_distance(T)
    for l in range(l_max - 1, 1, -1):
        for k in L[l]:
            if k not in T.nodes:
                continue
            chain = list(nx.all_simple_paths(T, k, core[0]))
            assert len(chain) == 1
            chain = chain[0]
            edges = [(chain[i - 1], chain[i]) for i in range(2, len(chain) - 1)]
            for e in edges:
                T.remove_edge(e[0], e[1])
                T1, T2 = [T.subgraph(c) for c in nx.connected_components(T)]
                if core[0] in T1.nodes:
                    T1, T2 = T2, T1
                best_edge = (e[0], e[1], G[e[0]][e[1]]['weight'])
                for t in G.edges(k):
                    if t[1] in T2.nodes:
                        T.add_weighted_edges_from([(k, t[1], G[k][t[1]]['weight'])])
                        d = average_pairwise_distance(T)
                        if d < min_cost:
                            min_cost = d
                            best_edge = (k, t[1], G[k][t[1]]['weight'])
                        T.remove_edge(k, t[1])
                T.add_weighted_edges_from([best_edge])
    return T


# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    G = read_input_file(path)
    T = solve(G)
    assert is_valid_network(G, T)
    print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    write_output_file(T, 'test.out')