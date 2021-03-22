import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
import torch


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    if torch.is_tensor(adj):
        adj = sp.coo_matrix(adj.cpu().detach().numpy())
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = (
        sp.eye(adj.shape[0])
        - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    )

    return torch.from_numpy(normalized_laplacian.astype(np.float32).todense())


def calculate_randomwalk_normalized_laplacian(adj):
    """
    # L = D^(-1)L
    # D = diag(A 1)
    :param adj:
    :return:
    """
    if torch.is_tensor(adj):
        adj = sp.coo_matrix(adj.cpu().detach().numpy())
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).tocoo()

    return torch.from_numpy(normalized_laplacian.astype(np.float32).todense())


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    adj_mx = adj_mx.cpu().detach().numpy()
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which="LM")
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format="csr", dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return torch.from_numpy(L.astype(np.float32).todense())


def cheb_polynomial(laplacian, cheb_k):
    """
    Compute the Chebyshev Polynomial, according to the graph laplacian.
    :param laplacian: the graph laplacian, [N, N].
    :return: the multi order Chebyshev laplacian, [K, N, N].
    """
    N = laplacian.size(0)  # [N, N]
    support_set = [
        torch.eye(N, device=laplacian.device, dtype=torch.float),
        laplacian,
    ]
    for k in range(2, cheb_k):
        support_set.append(torch.matmul(2 * laplacian, support_set[-1]) - support_set[-2])
    multi_order_laplacian = torch.stack(support_set, dim=0)
    return multi_order_laplacian
