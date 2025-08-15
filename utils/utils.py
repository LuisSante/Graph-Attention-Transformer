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

def get_graph_statistics(adj, features):
    num_nodes = adj.shape[0]
    num_edges = np.sum(adj) // 2 
    density = num_edges / (num_nodes * (num_nodes - 1) / 2)
    avg_degree = np.mean(np.sum(adj, axis=1))
    
    return {
        'nodes': num_nodes,
        'edges': int(num_edges),
        'density': density,
        'avg_degree': avg_degree,
        'features_shape': features.shape
    }