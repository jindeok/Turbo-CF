import numpy as np
import torch
from scipy.stats import rankdata
import copy
from scipy.linalg import expm


def recall_at_k(gt_mat, results, k=10):
    recall_sum = 0
    for i in range(gt_mat.shape[0]):
        relevant_items = set(np.where(gt_mat[i, :] > 0)[0])
        top_predicted_items = np.argsort(-results[i, :])[:k]
        num_relevant_items = len(relevant_items.intersection(top_predicted_items))
        recall_sum += num_relevant_items / len(relevant_items)
    recall = recall_sum / gt_mat.shape[0]
    return recall


def ndcg_at_k(gt_mat, results, k=10):
    ndcg_sum = 0
    for i in range(gt_mat.shape[0]):
        relevant_items = set(np.where(gt_mat[i, :] > 0)[0])
        top_predicted_items = np.argsort(-results[i, :])[:k]
        dcg = 0
        idcg = 0
        for j in range(k):
            if top_predicted_items[j] in relevant_items:
                dcg += 1 / np.log2(j + 2)
            if j < len(relevant_items):
                idcg += 1 / np.log2(j + 2)
        ndcg_sum += dcg / idcg if idcg > 0 else 0
    ndcg = ndcg_sum / gt_mat.shape[0]
    return ndcg


def calculate_row_correlations(matrix1, matrix2):
    base_value = 1  # 이거하고 랭크정규화 하는게 성능이 가장 좋네 230905 (0으로놓으면 폭망)

    num_rows = matrix1.shape[0]
    correlations = torch.zeros(num_rows)

    for row in range(num_rows):
        nz_indices1 = matrix1.indices[matrix1.indptr[row] : matrix1.indptr[row + 1]]
        nz_indices2 = matrix2.indices[matrix2.indptr[row] : matrix2.indptr[row + 1]]

        common_indices = torch.intersect1d(nz_indices1, nz_indices2)

        nz_values1 = matrix1.data[matrix1.indptr[row] : matrix1.indptr[row + 1]][
            torch.searchsorted(nz_indices1, common_indices)
        ]
        nz_values2 = matrix2.data[matrix2.indptr[row] : matrix2.indptr[row + 1]][
            torch.searchsorted(nz_indices2, common_indices)
        ]

        if len(common_indices) > 0:
            correlation = torch.corrcoef(nz_values1, nz_values2)[0, 1]
            correlations[row] = correlation + base_value

    return correlations


# 정규화
def corr_normalizer(corr_arr, version=0):
    # 랭크정규화
    if version == 0:
        ranks = np.apply_along_axis(rankdata, axis=1, arr=corr_arr)
        normalized_corr_arr = (corr_arr.shape[0] - ranks) / (corr_arr.shape[0])

    # Min-Max 정규화
    elif version == 1:
        # 각 행의 최솟값과 최댓값을 계산합니다.
        min_vals = np.min(corr_arr, axis=1, keepdims=True)
        max_vals = np.max(corr_arr, axis=1, keepdims=True)
        normalized_corr_arr = (corr_arr - min_vals) / (max_vals - min_vals)
        normalized_corr_arr[np.isnan(normalized_corr_arr)] = 0

    # 총합정규화
    # 각 행을 총합으로 나누어 총합이 1이 되도록 정규화합니다.
    elif version == 2:
        row_sums = np.sum(corr_arr, axis=1, keepdims=True)
        normalized_corr_arr = corr_arr / row_sums

    return normalized_corr_arr


def compute_dirichlet_energy(R, n_items):
    # 라플라시안 행렬 계산: L = D - A
    R = R.to_dense().T
    A = R @ R.T
    D = torch.diag(torch.sum(A, dim=1))  # 대각선 차수 행렬
    L = D - A  # 라플라시안 행렬

    # Dirichlet 에너지 계산: E(X) = sum(x_i^T L x_i)
    energy = torch.trace(torch.mm(torch.mm(R.t(), L), R))  # trace(X^T L X)
    return energy / n_items


def normalize_sparse_adjacency_matrix_(adj_matrix, alpha):
    # Calculate rowsum and columnsum using COO format
    rowsum = torch.sparse.mm(
        adj_matrix, torch.ones((adj_matrix.shape[1], 1), device=adj_matrix.device)
    ).squeeze()
    colsum = torch.sparse.mm(
        adj_matrix.t(), torch.ones((adj_matrix.shape[0], 1), device=adj_matrix.device)
    ).squeeze()

    # Calculate d_inv for rows and columns
    d_inv_rows = torch.pow(rowsum, -alpha)
    d_inv_rows[d_inv_rows == float("inf")] = 0.0
    d_mat_rows = torch.diag(d_inv_rows)

    d_inv_cols = torch.pow(colsum, alpha - 1)
    d_inv_cols[d_inv_cols == float("inf")] = 0.0
    d_mat_cols = torch.diag(d_inv_cols)

    # Normalize adjacency matrix
    norm_adj = d_mat_rows.mm(adj_matrix).mm(d_mat_cols)
    # norm_adj = d_mat_rows @ adj_matrix @ d_mat_cols
    return norm_adj


def csr2torch(csr_matrix):
    # Convert CSR matrix to COO format (Coordinate List)
    coo_matrix = csr_matrix.tocoo()

    # Create a PyTorch tensor for data, row indices, and column indices
    data = torch.FloatTensor(coo_matrix.data)
    # indices = torch.LongTensor([coo_matrix.row, coo_matrix.col])
    # -> This results in: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
    indices = torch.LongTensor(np.vstack((coo_matrix.row, coo_matrix.col)))

    # Create a sparse tensor using torch.sparse
    # return torch.sparse.FloatTensor(indices, data, torch.Size(coo_matrix.shape))
    # -> This results in: UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=).
    return torch.sparse_coo_tensor(indices, data, torch.Size(coo_matrix.shape))


def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized = (tensor - min_val) / (max_val - min_val)
    return normalized


def inference_3(MCEG, P, ps, cri, n_users, version=0, device="cpu"):
    MCEG_c = MCEG[n_users * cri : n_users * (cri + 1), :]
    P = P.to_dense()
    P_copy = copy.deepcopy(P)
    # linear
    if version == 0:
        P_copy.data **= ps[0]
        S = MCEG_c @ P_copy
    # Concave up
    elif version == 1:
        P_copy.data **= ps[1]
        S = MCEG_c @ (2 * P_copy - (P_copy @ P_copy))
    # Convex
    elif version == 2:
        P_copy.data **= ps[2]
        if device == "cpu":
            S = MCEG_c @ (expm(0.15 * P_copy))
        else:
            S = MCEG_c @ ((P_copy @ P_copy))
    elif version == 3:
        P_copy.data **= ps[3]
        mu = 0.5
        S = (
            MCEG_c
            @ torch.linalg.inv((P_copy + mu * torch.eye(len(P_copy), device=device)))
            @ P_copy.T
        )
    # Convex
    elif version == 4:
        P_copy.data **= ps[2]
        if device == "cpu":
            S = MCEG_c @ (expm(0.15 * P_copy))
        else:
            S = MCEG_c @ (
                torch.matrix_exp(
                    0.15 * (P_copy - torch.eye(len(P_copy), device=device))
                )
            )
    return S


def inference_4(MCEG, P, ps, cri, n_users, version=0, device="cpu"):
    MCEG_c = MCEG[n_users * cri : n_users * (cri + 1), :]
    P = P.to_dense()
    P_copy = copy.deepcopy(P)
    # linear
    if version == 0:
        P_copy.data **= ps[0]
        S = MCEG_c @ min_max_normalize(P_copy)
    # Concave up
    elif version == 1:
        P_copy.data **= ps[1]
        S = MCEG_c @ min_max_normalize(2 * P_copy - (P_copy @ P_copy))
    # Convex
    elif version == 2:
        P_copy.data **= ps[2]
        if device == "cpu":
            S = MCEG_c @ (expm(0.15 * P_copy))
        else:
            S = MCEG_c @ min_max_normalize(
                torch.matrix_exp(
                    0.15 * (P_copy - torch.eye(len(P_copy), device=device))
                )
            )
    elif version == 3:
        P_copy.data **= ps[3]
        mu = 0.5
        S = (
            MCEG_c
            @ torch.linalg.inv((P_copy + mu * torch.eye(len(P_copy), device=device)))
            @ P_copy.T
        )

    return S


def inference_4(MCEG, P, a, b, cri, n_users, version=0, device="cpu"):
    MCEG_c = MCEG[n_users * cri : n_users * (cri + 1), :]
    P = P.to_dense()
    P_2 = P @ P

    P_copy = copy.deepcopy(P)
    P_2_copy = copy.deepcopy(P_2)
    P_copy.data **= a
    P_2_copy.data **= b

    # linear
    if version == 0:
        S = MCEG_c @ P_copy
    # Concave up
    elif version == 1:
        S = MCEG_c @ (2 * P_copy - P_2_copy)
    return S


def normalize_sparse_adjacency_matrix_row(adj_matrix, alpha):
    rowsum = torch.sparse.mm(
        adj_matrix, torch.ones((adj_matrix.shape[1], 1), device=adj_matrix.device)
    ).squeeze()
    indices = (
        torch.arange(0, rowsum.size(0)).unsqueeze(1).repeat(1, 2).to(adj_matrix.device)
    )
    rowsum = torch.pow(rowsum, -1)
    d_mat_rows = torch.sparse.FloatTensor(
        indices.t(), rowsum, torch.Size([rowsum.size(0), rowsum.size(0)])
    ).to(device=adj_matrix.device)
    norm_adj = d_mat_rows.mm(adj_matrix)
    return norm_adj


def normalize_sparse_adjacency_matrix(adj_matrix, alpha):
    # Calculate rowsum and columnsum using COO format
    rowsum = torch.sparse.mm(
        adj_matrix, torch.ones((adj_matrix.shape[1], 1), device=adj_matrix.device)
    ).squeeze()
    rowsum = torch.pow(rowsum, -alpha)
    colsum = torch.sparse.mm(
        adj_matrix.t(), torch.ones((adj_matrix.shape[0], 1), device=adj_matrix.device)
    ).squeeze()
    colsum = torch.pow(colsum, alpha - 1)
    indices = (
        torch.arange(0, rowsum.size(0)).unsqueeze(1).repeat(1, 2).to(adj_matrix.device)
    )
    # d_mat_rows = torch.sparse.FloatTensor(
    #     indices.t(), rowsum, torch.Size([rowsum.size(0), rowsum.size(0)])
    # ).to(device=adj_matrix.device)
    # -> This results in:  UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=).
    d_mat_rows = torch.sparse_coo_tensor(
        indices.t(), rowsum, torch.Size([rowsum.size(0), rowsum.size(0)])
    ).to(device=adj_matrix.device)
    indices = (
        torch.arange(0, colsum.size(0)).unsqueeze(1).repeat(1, 2).to(adj_matrix.device)
    )
    # d_mat_cols = torch.sparse.FloatTensor(
    # indices.t(), colsum, torch.Size([colsum.size(0), colsum.size(0)])
    # ).to(device=adj_matrix.device)
    # -> This results in: UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=).
    d_mat_cols = torch.sparse_coo_tensor(
        indices.t(), colsum, torch.Size([colsum.size(0), colsum.size(0)])
    ).to(device=adj_matrix.device)

    # Calculate d_inv for rows and columns
    # d_inv_rows = torch.pow(rowsum, -alpha)
    # d_inv_rows[d_inv_rows == float('inf')] = 0.
    # d_mat_rows = torch.diag(d_inv_rows)

    # d_inv_cols = torch.pow(colsum, alpha - 1)
    # d_inv_cols[d_inv_cols == float('inf')] = 0.
    # d_mat_cols = torch.diag(d_inv_cols)

    # Normalize adjacency matrix
    norm_adj = d_mat_rows.mm(adj_matrix).mm(d_mat_cols)

    return norm_adj


def normalize_sparse_adjacency_matrix_general(adj_matrix, alpha, beta):
    """
    Identical to normalize_sparse_adjacency_matrix when 1 - alpha == beta
    """

    # Calculate rowsum and columnsum using COO format
    rowsum = torch.sparse.mm(
        adj_matrix, torch.ones((adj_matrix.shape[1], 1), device=adj_matrix.device)
    ).squeeze()
    rowsum = torch.pow(rowsum, -alpha)
    colsum = torch.sparse.mm(
        adj_matrix.t(), torch.ones((adj_matrix.shape[0], 1), device=adj_matrix.device)
    ).squeeze()
    colsum = torch.pow(colsum, -beta)
    indices = (
        torch.arange(0, rowsum.size(0)).unsqueeze(1).repeat(1, 2).to(adj_matrix.device)
    )
    d_mat_rows = torch.sparse.FloatTensor(
        indices.t(), rowsum, torch.Size([rowsum.size(0), rowsum.size(0)])
    ).to(device=adj_matrix.device)
    indices = (
        torch.arange(0, colsum.size(0)).unsqueeze(1).repeat(1, 2).to(adj_matrix.device)
    )
    d_mat_cols = torch.sparse.FloatTensor(
        indices.t(), colsum, torch.Size([colsum.size(0), colsum.size(0)])
    ).to(device=adj_matrix.device)

    # Normalize adjacency matrix
    norm_adj = d_mat_rows.mm(adj_matrix).mm(d_mat_cols)

    return norm_adj


def graph_construction(R_tr, n_cri, device, version=0):
    # Single graph w/ overall ratings
    if version == 0:
        MCEG = R_tr[0]

    # MC Expansion graph construction (ablation with uniform)
    elif version == 1:
        R_tr_dense = []
        R_tr_dense.append(R_tr[0].to_dense())
        for i in range(n_cri - 1):
            R_tr[i + 1] = R_tr[i + 1].to_dense()
            R_tr_dense.append(R_tr[i + 1])
        MCEG = torch.hstack(R_tr_dense)
        MCEG = MCEG.to_sparse_csr()

    # MC Expansion graph construction (ablation with uniform)
    elif version == 2:
        R_tr_dense = []
        R_tr_dense.append(R_tr[0].to_dense())
        for i in range(n_cri - 1):
            R_tr[i + 1] = R_tr[i + 1].to_dense()
            R_tr_dense.append(R_tr[i + 1])
        MCEG = torch.vstack(R_tr_dense)
        # MCEG = MCEG.to_sparse_csr()
        MCEG = MCEG.to_sparse()

    return MCEG.to(device)


def freq_filter(s_values, mode=1, alpha=0.9, start=0):
    """
    input:
    - s_values: singular (eigen) values, list form

    output:
    - filterd_s_values
    """
    if mode == 0:
        filtered_s_values = s_values
    elif mode == 1:
        filtered_s_values = [(lambda x: 1 / (1 - alpha * x))(v) for v in s_values]
    elif mode == 2:
        filtered_s_values = [(lambda x: 1 / (alpha * x))(v) for v in s_values]
    elif mode == 3:
        filtered_s_values = [(lambda x: 1.5**x)(v) for v in s_values]
    elif mode == 3:
        filtered_s_values = [(lambda x: 1.5**x)(v) for v in s_values]
    elif mode == "band_pass":
        end = start + 5
        filtered_s_values = (
            [0] * int(start) + [1] * int(end - start) + [0] * int(len(s_values) - end)
        )

    return np.diag(filtered_s_values)


def get_norm_adj(alpha, adj_mat):
    # Calculate rowsum and columnsum using PyTorch operations
    rowsum = torch.sum(adj_mat, dim=1)
    colsum = torch.sum(adj_mat, dim=0)

    # Calculate d_inv for rows and columns
    d_inv_rows = torch.pow(rowsum, -alpha).flatten()
    d_inv_rows[torch.isinf(d_inv_rows)] = 0.0
    d_mat_rows = torch.diag(d_inv_rows)

    d_inv_cols = torch.pow(colsum, alpha - 1).flatten()
    d_inv_cols[torch.isinf(d_inv_cols)] = 0.0
    d_mat_cols = torch.diag(d_inv_cols)
    d_mat_i_inv_cols = torch.diag(1 / d_inv_cols)

    # Normalize adjacency matrix
    norm_adj = adj_mat.mm(d_mat_rows).mm(adj_mat).mm(d_mat_cols)
    norm_adj = norm_adj.to_sparse()  # Convert to sparse tensor

    # Convert d_mat_rows, d_mat_i_inv_cols to sparse tensors
    d_mat_rows_sparse = d_mat_rows.to_sparse()
    d_mat_i_inv_cols_sparse = d_mat_i_inv_cols.to_sparse()

    return norm_adj


# Example usage
# alpha = ...
# adj_mat = ...
# norm_adj, d_mat_rows, d_mat_i_inv_cols = get_norm_adj(alpha, adj_mat)


# Evaluation functions


def top_k(S, k=1, device="cpu"):
    """
    S: scores, numpy array of shape (M, N) where M is the number of source nodes,
        N is the number of target nodes
    k: number of predicted elements to return
    """
    if device == "cpu":
        top = torch.argsort(-S)[:, :k]
        result = torch.zeros(S.shape)
        for idx, target_elms in enumerate(top):
            for elm in target_elms:
                result[idx, elm] = 1
    else:
        top = torch.argsort(-S)[:, :k]
        result = torch.zeros(S.shape, device=device)
        for idx, target_elms in enumerate(top):
            for elm in target_elms:
                result[idx, elm] = 1
    return result, top


def precision_k_(topk, gt, k, device="cpu"):
    """
    topk, gt: (UXI) array
    k: @k measurement
    """
    return (
        np.multiply(topk, gt).sum() / (k * len(gt))
        if device == "cpu"
        else torch.mul(topk, gt).sum() / (k * len(gt))
    )


def precision_k(topk, gt, k, device="cpu"):
    """
    topk, gt: (U, X, I) array, where U is the number of users, X is the number of items per user, and I is the number of items in total.
    k: @k measurement
    """
    if device == "cpu":
        precision_values = []
        for i in range(topk.shape[0]):
            num_correct = np.multiply(topk[i], gt[i]).sum()
            precision_i = num_correct / (k)
            precision_values.append(precision_i)
        mean_precision = np.mean(precision_values)
    else:
        precision_values = []
        for i in range(topk.shape[0]):
            num_correct = torch.mul(topk[i], gt[i]).sum()
            precision_i = num_correct / (k)
            precision_values.append(precision_i)
        mean_precision = torch.mean(torch.tensor(precision_values))

    return mean_precision


def recall_k_(topk, gt, k, device="cpu"):
    """
    topk, gt: (UXI) array
    k: @k measurement
    """
    return (
        np.multiply(topk, gt).sum() / gt.sum()
        if device == "cpu"
        else torch.mul(topk, gt).sum() / gt.sum()
    )


def recall_k(topk, gt, k, device="cpu"):
    """
    topk, gt: (U, X, I) array, where U is the number of users, X is the number of items per user, and I is the number of items in total.
    k: @k measurement
    """
    if device == "cpu":
        recall_values = []
        for i in range(topk.shape[0]):
            recall_i = (
                np.multiply(topk[i], gt[i]).sum() / gt[i].sum()
                if gt[i].sum() != 0
                else 0
            )
            if gt[i].sum() != 0:
                recall_values.append(recall_i)
        mean_recall = np.mean(recall_values)
    else:
        recall_values = []
        for i in range(topk.shape[0]):
            recall_i = (
                torch.mul(topk[i], gt[i]).sum() / gt[i].sum() if gt[i].sum() != 0 else 0
            )
            if gt[i].sum() != 0:
                recall_values.append(recall_i)
        mean_recall = torch.mean(torch.tensor(recall_values))

    return mean_recall


def ndcg_k(rels, rels_ideal, gt, device="cpu"):
    """
    rels: sorted top-k arr
    rels_ideal: sorted top-k ideal arr
    """
    k = rels.shape[1]
    n = rels.shape[0]

    ndcg_values = []
    for row in range(n):
        dcg = 0
        idcg = 0
        for col in range(k):
            if gt[row, rels[row, col]] == 1:
                if col == 0:
                    dcg += 1
                else:
                    dcg += 1 / np.log2(col + 1)
            if gt[row, rels_ideal[row, col]] == 1:
                if col == 0:
                    idcg += 1
                else:
                    idcg += 1 / np.log2(col + 1)
        if idcg != 0:
            ndcg_values.append(dcg / idcg)

    mean_ndcg = torch.mean(torch.tensor(ndcg_values))

    return mean_ndcg
