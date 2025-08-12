import numpy as np
from config import LARGE_NEGATIVE_NUMBER

# xavier initialization for weights
def init_weights(input_dim,  output_dim, seed=None):
    rng = np.random.RandomState(seed)
    limit = np.sqrt(6.0 / (input_dim + output_dim))
    return rng.uniform(-limit, limit, (input_dim, output_dim)).astype(float)

def leaky_relu(arr, alpha=0.2):
    return np.where(arr > 0, arr, alpha * arr)

def softmax_with_mask(x, mask, axis=1):
    x_masked = np.where(mask, x, LARGE_NEGATIVE_NUMBER)
    x_shifted = x_masked - np.max(x_masked, axis=axis, keepdims=True)
    exp = np.exp(x_shifted)
    exp = exp * mask  # anulate non-neighbors
    denom = np.sum(exp, axis=axis, keepdims=True) + 1e-12
    return exp / denom