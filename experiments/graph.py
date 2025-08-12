import numpy as np

from  utils.utils import init_weights, leaky_relu, softmax_with_mask
from data.dataset import create_graph, adj_to_edge_index, edge_index_to_adj

print("==============================================================================")
print("Test init weights")
W = init_weights(5, 3, seed=42)

print("\n Test leaky_relu ")
arr = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
print("Input:", arr)
print("Output:", leaky_relu(arr, alpha=0.1))

print("\n Test softmax_with_mask ")
logits = np.array([[1.0, 2.0, 3.0],
                   [3.0, 2.0, 1.0]])
mask = np.array([[True, True, False],
                 [True, False, True]])
print("Logits:\n", logits)
print("Mask:\n", mask)
print("Softmax masked:\n", softmax_with_mask(logits, mask))

print("==============================================================================")
print("Test create_graph")
adj, feats, labels = create_graph()
print("Adj:\n", adj)
print("Features shape:", feats.shape)
print("Labels:", labels)

ei = adj_to_edge_index(adj)
print("\nEdge index:\n", ei)

adj_back = edge_index_to_adj(ei, num_nodes=adj.shape[0])
print("\nAdj reconstruida:\n", adj_back)
print("==============================================================================")
