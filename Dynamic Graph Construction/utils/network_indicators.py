import networkx as nx


def calculate_modularity(G, target, class_num=5):  # absolutely sure
    r'''

    Args:
        G: undirected weighted graph
        target: category of nodes, like [[1, 3], [0, 2, 4]] or [{1, 3}, {0, 2, 4}]
        class_num: the number of class images

    Returns: modularity

    '''
    node_category = []
    # deal with nodes' category
    for i in range(class_num):
        a = [j for j, x in enumerate(target) if x == i]
        node_category.append(a)
        # node 0 to batch_size, like
        # [[0, 4, 13, 19, 20, 29], [5, 6, 7, 12, 16, 17], [8, 9, 22, 23, 24, 27], [10, 11, 15, 18, 21, 25], [1, 2, 3, 14, 26, 28]]

    modularity = nx.algorithms.community.quality.modularity(G, node_category)
    return modularity
















