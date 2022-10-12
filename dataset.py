""" Code adapted from https://github.com/kavehhassani/mvgrl """
import numpy as np
import torch as th
import scipy.sparse as sp
from scipy.linalg import fractional_matrix_power, inv
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import networkx as nx
from sklearn.preprocessing import MinMaxScaler


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense(), sparse_to_tuple(features)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    """Calculate Diffusion Matrix"""
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])  # A^ = A + I_n
    d = np.diag(np.sum(a, 1))  # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)  # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))  # a(I_n-(1-a)A~)^-1


def process_dataset(name):
    if name == 'Cora':
        dataset = CoraGraphDataset(raw_dir='./')
    elif name == 'CiteSeer':
        dataset = CiteseerGraphDataset(raw_dir='./')
    elif name == 'PubMed':
        dataset = PubmedGraphDataset(raw_dir='./')

    graph = dataset[0]
    feat = graph.ndata.pop('feat')
    label = graph.ndata.pop('label')

    train_mask = graph.ndata.pop('train_mask')
    val_mask = graph.ndata.pop('val_mask')
    test_mask = graph.ndata.pop('test_mask')

    nx_g = dgl.to_networkx(graph)
    diff_adj = compute_ppr(nx_g, 0.2)

    if name == 'PubMed':
        feat = th.tensor(preprocess_features(feat.numpy())).float()
        epsilons = [1e-5]
        adj = nx.convert_matrix.to_numpy_array(nx_g)
        avg_degree = np.sum(adj) / adj.shape[0]
        epsilon = epsilons[np.argmin([abs(avg_degree - np.argwhere(diff_adj >= e).shape[0] / diff_adj.shape[0])
                                      for e in epsilons])]
        diff_adj[diff_adj < epsilon] = 0
        scaler = MinMaxScaler()
        scaler.fit(diff_adj)
        diff_adj = scaler.transform(diff_adj)

    diff_edges = np.nonzero(diff_adj)
    diff_weight = diff_adj[diff_edges]
    diff_graph = dgl.graph(diff_edges)

    graph = graph.add_self_loop()

    return graph, diff_graph, feat, label, train_mask, val_mask, test_mask, diff_weight