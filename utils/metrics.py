import numpy as np

def accuracy(predictions, targets):
    return np.mean(predictions == targets)

def micro_f1_score(predictions, targets):
    pass