from parse import *
import networkx as nx
import solver_hl
import os

if __name__ == "__main__":
    output_dir = "out/"
    input_dir = "inputs"
    failed_path = "./failed.txt"
    remaining = "./remaining_files.txt"
    files = []
    with open(remaining, 'r') as f:
        for i in range(59):
            files.append(f.readline().strip())
    for input_path in os.listdir(input_dir):
        graph_name = input_path.split(".")[0]
        if input_path in files:
                #print("input path: {}".format(input_path))
                #print("graph name: {}".format(graph_name))
                out = output_dir + input_path.replace('.in', '.out')
                print("output dir: {}".format(out))
                G = read_input_file("{}/{}".format(input_dir, input_path))
                print("starting: {}".format(graph_name))
                try:
                    T = solver_hl.solve(G)
                    write_output_file(T, out)
                    print("solved: {}".format(input_path))
                except:
                    print(graph_name)
                    with open(failed_path, 'a') as f:
                        f.writelines([graph_name])
                        f.writelines("\n")
