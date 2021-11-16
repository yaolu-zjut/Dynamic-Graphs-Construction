import heapq
import torch
import dgl.backend as B
import numpy as np
from scipy.stats import stats
import torch.nn.functional as F


def calculate_similarity(feat, topk, similarity_function='Pearson'):
    r'''

    Args:
        feat: features extracted by pretrained CNN
        topk: topk samples
        similarity_function: which indicators to use

    Returns: similarity_matrix, edges_list

    '''
    edges_list = []
    n_samples = feat.shape[0]
    similarity_matrix = np.zeros((n_samples, n_samples))
    # similarity_matrix -= np.eye(n_samples)  # to avoid edges less than expectation for using Pearson

    for i in range(n_samples):
        for j in range(n_samples):
            if i < j:
                if similarity_function == 'cosine':  # already change to gpu
                    similarity = Cosine_Similarity(feat[i].reshape(1, -1), feat[j].reshape(1, -1))
                elif similarity_function == 'Pearson':  # note that use Pearson will make edges less than expectation
                    similarity, _ = stats.pearsonr(feat[i].cpu().numpy(), feat[j].cpu().numpy())  # not right

                similarity_matrix[i][j] = similarity_matrix[j][i] = similarity

    for i in range(n_samples):  # note that: nodes start from 1
        k_indice = heapq.nlargest(topk, range(len(similarity_matrix[i])), similarity_matrix[i].take)
        for j in range(len(k_indice)):
            b = int(k_indice[j]+1)
            a = (int(i+1), b, float(similarity_matrix[i][k_indice[j]]))
            edges_list.append(a)

    return similarity_matrix, edges_list


def calculate_distance_to_similarity(feat, topk, similarity_function='cos_distance_softmax'):
    r'''

    Args:
        feat: features extracted by pretrained CNN
        topk: topk samples
        similarity_function: which indicators to use

    Returns: distance_matrix, edges_list
    distance_matrix like that:
    tensor([[-1.0000,  0.2970,  0.4616,  0.7445,  0.2955],
        [ 0.2970, -1.0000,  0.1565,  0.6804,  0.1723],
        [ 0.4616,  0.1565, -1.0000,  0.8887,  0.3414],
        [ 0.7445,  0.6804,  0.8887, -1.0000,  0.8084],
        [ 0.2955,  0.1723,  0.3414,  0.8084, -1.0000]], device='cuda:0')
    '''
    edges_list = []
    distance = 0
    n_samples = feat.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    if similarity_function == 'cos_distance_softmax':
        distance_matrix = cos_distance_softmax(feat)  # using matrix multiplication
        fil = 1 - torch.eye(n_samples, n_samples)
        distance_matrix = distance_matrix * fil.cuda()
        distance_matrix = distance_matrix - torch.eye(n_samples, n_samples).cuda()
        # noted that we would delete the self-loop, so topk needs to increase 1
        # noted that here the distance smaller , better
        k_indices = B.argtopk(distance_matrix, topk+1, 1, descending=False)
    # elif similarity_function == 'euclidean':  # not finished
    #     for i in range(n_samples):
    #         for j in range(n_samples):
    #             if i < j:
    #                 distance = F.pairwise_distance(feat[i].reshape(1, -1), feat[j].reshape(1, -1), p=2)
    #
    #             distance_matrix[i][j] = distance_matrix[j][i] = distance
    #     print(distance_matrix)
    #     distance_matrix = np.linalg.norm(distance_matrix, keepdims=True)
    #     print(distance_matrix)

    for i in range(n_samples):  # note that: nodes start from 1
        for j in range(k_indices.shape[1]):
            b = int(k_indices[i, j]+1)
            if i+1 != b:  # delete self-loop
                a = (int(i+1), b, float(distance_matrix[i][k_indices[i, j]]))
                edges_list.append(a)
    return distance_matrix, edges_list


def cos_distance_softmax(x, eps = 1e-7):
    r'''

    Args:
        x: torch.tensor(gpu), will be 2 dimension
        eps: avoid to divide 0

    Returns: Similarity matrix

    '''
    soft = F.softmax(x, dim=1)
    w = soft.norm(p=2, dim=1, keepdim=True)
    return 1 - soft @ soft.T / (w @ w.T).clamp(min=eps)


def Cosine_Similarity(x, y):
    r'''

    Args:
        x: feature
        y: feature

    Returns: the similarity between x and y

    '''

    return torch.cosine_similarity(x, y)



