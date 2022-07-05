import networkx as nx
from args import args
import numpy as np


def directly_return_undirected_weighted_network(edges_list):  # checked
    r'''

    Args:
        edges_list: direcetd weighted networks' edges list, a tuple like: (source node, target node, weight)

    Returns: undirected weighted network, the adj of undirected weighted network

    '''
    # print('edges_list:', edges_list)
    adj = np.zeros((args.batch_size, args.batch_size))
    num = args.batch_size

    # directed adj
    for ii in range(len(edges_list)):
        a, b = edges_list[ii][0] - 1, edges_list[ii][1] - 1
        adj[a, b] = edges_list[ii][2]

    print('directed adj:', adj)
    # undirected adj
    adj = (adj + adj.T) / 2
    print('undirected adj:', adj)
    undirected_weighted_network = nx.from_numpy_matrix(adj)
    return undirected_weighted_network, adj


