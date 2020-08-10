import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import sys
import numpy as np
import solver
import solver1
import solver2


def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        solution tree
    """

    T1 = solver.solve(G)
    T2 = solver1.solve(G)
    T3 = solver2.solve(G)

    if is_valid_network(G, T1) != True: 
    	return min([T2, T3], key = lambda x: average_pairwise_distance(x))
    elif is_valid_network(G, T2) != True: 
        return min([T1, T3], key = lambda x: average_pairwise_distance(x))
    elif is_valid_network(G, T3) != True: 
        return min([T1, T2], key = lambda x: average_pairwise_distance(x))
    else:
    	return min([T1, T2, T3], key = lambda x: average_pairwise_distance(x))


if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    G = read_input_file(path)
    T = solve(G)
    assert is_valid_network(G, T)
    out = path.replace('inputs', 'out').replace('.in', '.out') 
    print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    write_output_file(T, out)