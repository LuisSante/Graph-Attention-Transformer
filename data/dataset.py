import numpy as np
from config import Config

# Create a graph with 4 nodes (adjaceny matrix, edge list, self-loops)
def create_graph(num_nodes=4, num_features=5, num_classes=3, self_loops=False, seed=Config.SEED):
    rng = np.random.RandomState(seed)

    p = 0.3
    adj_upper = rng.rand(num_nodes, num_nodes) < p
    adj = np.triu(adj_upper, k=1)  
    adj = adj + adj.T              

    adj = adj.astype(np.float32)

    if self_loops:
        np.fill_diagonal(adj, 1.0)

    features = rng.randn(num_nodes, num_features).astype(np.float32)
    labels = rng.randint(0, num_classes, size=num_nodes, dtype=np.int64)

    return adj, features, labels

def add_self_loops(adj):
    adj_with_loops = adj.copy()
    np.fill_diagonal(adj_with_loops, 1.0)
    return adj_with_loops


def adj_to_edge_index(adj):
    src, dst = np.nonzero(adj)
    return np.vstack((src, dst))


def edge_index_to_adj(edge_index, num_nodes):
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    src, dst = edge_index
    adj[src, dst] = 1.0
    return adj