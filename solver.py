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

    # TODO: your code here!
    subsets = preprocess(G)
    try:
        tree = greedy_dominating_tree(G, subsets)
        DT = postprocess(G, tree)
    except:
        DT = nx.minimum_spanning_tree(G)
    return DT


def preprocess(G):
    '''
    Preprocess a graph into a dominating set problem by generating all the subsets
    based on adjacent vertices
    '''
    subsets = {}
    for n, nbrs in G.adj.items():
        strings = np.append(list(nbrs.keys()), n)
        subsets[n] = strings
    return subsets


def greedy_dominating_tree(G, subsets):
    '''
    This is a greedy algorithm to find the smallest dominating tree of the Graph G
    This algorithm doesn't yet minimize pair wise distance.
    '''
    tree = []
    dominated = np.array([])
    while len(dominated) < len(list(G.nodes())):
        best_node = -1
        best_dominating_set = []
        tree_nbrs = get_nodes_adj_tree(G, tree)
        for node in tree_nbrs:
            dominating_set = delta_dominating(dominated, subsets[node])
            if (len(dominating_set) > len(best_dominating_set)):
                #print("current best dominating: {}".format(dominating_set))
                best_node = node
                best_dominating_set = dominating_set

        #print("best dominating node: {}, set: {}".format(best_node,best_dominating_set))
        tree.append(best_node)
        dominated = np.append(dominated, best_dominating_set)
        dominated.flatten()
        dominated = np.unique(dominated)
        #print("dominated: {}".format(dominated))
    return tree

def get_nodes_adj_tree(G, nodes):
    if (len(nodes) == 0):
        return list(G.nodes())
    nbrs = np.array([])
    for node in nodes:
        nbrs = np.append(nbrs, list(G.neighbors(node)))
    nbrs.flatten()
    unique, counts = np.unique(nbrs, return_counts = True)
    #print(unique)
    #print(counts)
    return [unique[i] for i in range(len(unique)) if counts[i] == 1]

def intersection_size(lst1, lst2):
    temp = set(lst2)
    return len([value for value in lst1 if value in temp])

def delta_dominating(curr_dominated, potential_addition):
    return [elem for elem in potential_addition if elem not in curr_dominated]

def postprocess(G, tree):
    '''
    Given a list of nodes in G (tree), create an nx Tree based on the edges in G
    '''
    #print("\n\n\n\nIn postprocess.")
    Tree = G.copy()
    #print("total nodes: {}".format(list(Tree.nodes)))
    #print("tree ndoes: {}".format(tree))
    nodes_to_remove = [node for node in list(Tree.nodes) if node not in tree]
    for node in nodes_to_remove:
        Tree.remove_node(node)
    return Tree

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
