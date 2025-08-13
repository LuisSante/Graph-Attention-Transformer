import numpy as np
from utils.utils import softmax, sigmoid

def cross_entropy_loss(logits, targets, mask=None):
    if targets.ndim == 1:
        num_classes = logits.shape[1]
        targets_onehot = np.eye(num_classes)[targets]
    else:
        targets_onehot = targets
    
    probs = softmax(logits, axis=1)
    
    epsilon = 1e-12  # For numerical stability
    probs = np.clip(probs, epsilon, 1 - epsilon)
    loss = -np.sum(targets_onehot * np.log(probs), axis=1)
    
    if mask is not None:
        loss = loss * mask
        return np.sum(loss) / np.sum(mask)
    
    return np.mean(loss)

def binary_cross_entropy_loss(logits, targets, mask=None):
    probs = sigmoid(logits)
    epsilon = 1e-12
    probs = np.clip(probs, epsilon, 1 - epsilon)
    
    loss = -(targets * np.log(probs) + (1 - targets) * np.log(1 - probs))
    loss = np.sum(loss, axis=1)
    
    if mask is not None:
        loss = loss * mask
        return np.sum(loss) / np.sum(mask)
    
    return np.mean(loss)