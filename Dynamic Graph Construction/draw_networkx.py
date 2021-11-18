import itertools
import networkx as nx
import matplotlib.pyplot as plt
from args import args
import numpy as np


def directly_draw_undirected_weighted_network(edges_list, whole_label, target, layer, number_of_class=5, color=['r', 'b', 'y', 'g', 'purple']):  # checked
    r'''

    Args:
        edges_list: direcetd weighted networks' edges list, a tuple like: (source node, target node, weight)
        whole_label: a list of all samples' ground true & predicted top5 labels
        target: ground true label
        layer: draw which layer
        number_of_class: number of classes
        color: a list of colors to draw the nodes

    Returns: undirected weighted network, the adj of undirected weighted network

    '''
    plt.figure(layer)
    color_list = []
    edge_weight = []
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
    # undirected_weighted_network is from 0 to batch_size -1 !!!
    undirected_weighted_network = nx.from_numpy_matrix(adj)
    pos = nx.circular_layout(undirected_weighted_network)

    # deal with edges' color
    _, weights = zip(*nx.get_edge_attributes(undirected_weighted_network, 'weight').items())  # have problem for Iresnet152 small_ImageNet
    weights = list(weights)

    # deal with nodes' color
    for i in range(number_of_class):
        a = [j for j, x in enumerate(target) if x == i]
        color_list.append(a)
        # node 0 to batch_size, like
        # [[0, 4, 13, 19, 20, 29], [5, 6, 7, 12, 16, 17], [8, 9, 22, 23, 24, 27], [10, 11, 15, 18, 21, 25], [1, 2, 3, 14, 26, 28]]

    # draw colorful nodes
    for i in range(len(color_list)):
        nx.draw_networkx_nodes(undirected_weighted_network, pos, nodelist=color_list[i], node_color=color[i], node_size=20)

    whole_label = whole_label.cpu().numpy()
    for i in range(args.batch_size):
        undirected_weighted_network.nodes[i]['label'] = whole_label[i]

    # draw colorful edges
    edges = nx.draw_networkx_edges(undirected_weighted_network, pos=pos, arrowstyle='->', edge_color=weights, edge_cmap=plt.cm.Oranges)
    # pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Oranges)  # NOT right
    # plt.colorbar(pc)

    # draw colorful labels
    # node_data = nx.get_node_attributes(undirected_weighted_network, 'label')
    # nx.draw_networkx_labels(undirected_weighted_network, pos, labels=node_data, font_color="brown", font_size=8)
    plt.title('layer %d' % layer)
    plt.savefig('save_fig/%s_%s_undirecetd_weighted_network_%d.jpg' % (args.set, args.arch, layer))

    return undirected_weighted_network, adj


def draw_ground_true_graph(target):  # checked
    nodes = []
    G = nx.Graph()
    target.cpu().numpy().tolist()
    for i in range(10):
        nodes.append([j for j, x in enumerate(target) if x == i])
        nodes[i] = (np.array(nodes[i]) + 1).tolist()
        G.add_nodes_from(nodes[i], label=i)
        edges = list(itertools.combinations(nodes[i], 2))  # connect two nodes with same labels
        G.add_edges_from(edges)

    pos=nx.circular_layout(G)
    nx.draw(G, pos, node_size=100, with_labels=True, font_size=6)
    plt.savefig('true_label.jpg')


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


