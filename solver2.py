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
    T = MST_pruned(G)
    return T

def MST_pruned(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        A networkx.Graph
    """
    T = nx.minimum_spanning_tree(G)
    leaves = [node for node in T.nodes if T.degree[node] == 1]
    if len(leaves) != len(T.nodes):
        T.remove_nodes_from(leaves)
    return T

if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    G = read_input_file(path)
    T = solve(G)
    assert is_valid_network(G, T)
    print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    write_output_file(T, 'test.out')
