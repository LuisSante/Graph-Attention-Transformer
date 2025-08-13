import numpy as np
from config import LARGE_NEGATIVE_NUMBER

# xavier initialization for weights
def init_weights(input_dim,  output_dim, seed=None):
    rng = np.random.RandomState(seed)
    limit = np.sqrt(6.0 / (input_dim + output_dim))
    return rng.uniform(-limit, limit, (input_dim, output_dim)).astype(float)

def leaky_relu(arr, alpha=0.2):
    return np.where(arr > 0, arr, alpha * arr)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def relu(x):
    return np.maximum(0, x)

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def sigmoid(x):
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))

def create_mask_from_adjacency(adj):
    return adj != 0

def apply_dropout(x, dropout_rate, training=True, seed=None):
    if not training or dropout_rate == 0:
        return x
    
    rng = np.random.RandomState(seed)
    keep_prob = 1 - dropout_rate
    mask = rng.rand(*x.shape) < keep_prob
    
    return (x * mask) / keep_prob

def l2_regularization(weights_list, lambda_reg):
    l2_loss = 0
    for weights in weights_list:
        l2_loss += np.sum(weights ** 2)
    return lambda_reg * l2_loss


def softmax_with_mask(x, mask, axis=1):
    x_masked = np.where(mask, x, LARGE_NEGATIVE_NUMBER)
    x_shifted = x_masked - np.max(x_masked, axis=axis, keepdims=True)
    exp = np.exp(x_shifted)
    exp = exp * mask  # anulate non-neighbors
    denom = np.sum(exp, axis=axis, keepdims=True) + 1e-12
    return exp / denom

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)