import numpy as np
from config import Config

# xavier initialization for weights
def init_weights(input_dim,  output_dim, seed=None):
    rng = np.random.RandomState(seed)
    limit = np.sqrt(6.0 / (input_dim + output_dim))
    weights = rng.uniform(-limit, limit, (input_dim, output_dim)).astype(float)
    print(f"    Initializing weights: {weights.shape} with limit ±{limit:.4f}")
    return weights

def leaky_relu(arr, alpha=0.2):
    result = np.where(arr > 0, arr, alpha * arr)
    print(f"LeakyReLU applied (α={alpha}): min={result.min():.4f}, max={result.max():.4f}")
    return result

def softmax_with_mask(x, mask, axis=1):
    print(f"Applying softmax with mask...") 
    print(f"Input shape: {x.shape}, Active mask: {mask.sum()} elements")

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
        in_degrees = np.sum(adj, axis=0)
        out_degrees = np.sum(adj, axis=1)
        
        stats.update({
            "Average in-degree": f"{np.mean(in_degrees):.2f}",
            "Average out-degree": f"{np.mean(out_degrees):.2f}",
            "Max in-degree": int(np.max(in_degrees)),
            "Max out-degree": int(np.max(out_degrees)),
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