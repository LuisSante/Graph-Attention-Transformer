import numpy as np
from config import Config

def init_weights(input_dim, output_dim, seed=None):
    """Xavier initialization for weights"""
    rng = np.random.RandomState(seed)
    limit = np.sqrt(6.0 / (input_dim + output_dim))
    weights = rng.uniform(-limit, limit, (input_dim, output_dim)).astype(float)
    #print(f"    Initializing weights: {weights.shape} with limit ±{limit:.4f}")
    return weights

def leaky_relu(arr, alpha=0.2):
    result = np.where(arr > 0, arr, alpha * arr)
    print(f"LeakyReLU applied (α={alpha}): min={result.min():.4f}, max={result.max():.4f}")
    return result

def softmax_with_mask(x, mask, axis=1):
    print(f"      Applying softmax with mask...") 
    print(f"      Input shape: {x.shape}, Active mask: {mask.sum()} elements")

    x_masked = np.where(mask, x, Config.LARGE_NEGATIVE_NUMBER)
    x_shifted = x_masked - np.max(x_masked, axis=axis, keepdims=True)
    exp = np.exp(x_shifted)
    exp = exp * mask  # anulate non-neighbors
    denom = np.sum(exp, axis=axis, keepdims=True) + 1e-12
    return exp / denom

def get_graph_statistics(adj, features, directed=True):
    num_nodes = adj.shape[0]
    total_edges = int(np.sum(adj))
    
    stats = {
        "Nodes": num_nodes,
        "Edges": total_edges,
        "Graph type": "Directed" if directed else "Undirected",
        "Node features": features.shape[1],
        "Feature range": f"[{features.min():.3f}, {features.max():.3f}]"
    }
    
    if directed:
        stats.update({
            "Reciprocal edges": int(np.sum(adj * adj.T) / 2),
            "Density": f"{total_edges / (num_nodes * (num_nodes - 1)):.3f}"
        })
    else:
        degrees = np.sum(adj, axis=0)
        stats.update({
            "Average degree": f"{np.mean(degrees):.2f}",
            "Max degree": int(np.max(degrees)),
            "Density": f"{total_edges / (num_nodes * (num_nodes - 1) / 2):.3f}"
        })
    
    return stats

def get_batch_statistics(graphs_data, directed=True):
    n_graphs = len(graphs_data)
    densities = []
    
    for graph in graphs_data:
        num_nodes = graph['num_nodes']
        num_edges = int(np.sum(graph['adj']))
        if directed:
            possible_edges = num_nodes * (num_nodes - 1)
        else:
            possible_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max(possible_edges, 1)
        densities.append(density)
    
    stats = {
        "Number of graphs": n_graphs,
        "Density range": f"{min(densities):.3f}-{max(densities):.3f}",
        "Average density": f"{np.mean(densities):.3f}",
        "Graph type": "Directed" if directed else "Undirected",
        "Features per node": graphs_data[0]['features'].shape[1]
    }
    
    return stats