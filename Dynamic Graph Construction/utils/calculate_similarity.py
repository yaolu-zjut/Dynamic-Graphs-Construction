import heapq
import torch
import numpy as np

def calculate_cosine_similarity_matrix(h_emb, topk, eps=1e-8):
    r'''
        h_emb: (N, M) hidden representations
    '''
    # normalize
    edges_list = []
    a_n = h_emb.norm(dim=1).unsqueeze(1)
    a_norm = h_emb / torch.max(a_n, eps * torch.ones_like(a_n))

    # cosine similarity matrix
    sim_matrix = torch.einsum('bc,cd->bd', a_norm, a_norm.transpose(0,1))
    sim_matrix = sim_matrix.cpu().numpy()
    row, col = np.diag_indices_from(sim_matrix)  # Don not consider self-similarity
    sim_matrix[row, col] = 0
    n_samples = h_emb.shape[0]
    for i in range(n_samples):  # note that: nodes start from 1
        k_indice = heapq.nlargest(topk, range(len(sim_matrix[i])), sim_matrix[i].take)
        for j in range(len(k_indice)):
            b = int(k_indice[j]+1)
            a = (int(i+1), b, float(sim_matrix[i][k_indice[j]]))
            edges_list.append(a)
    return sim_matrix, edges_list





