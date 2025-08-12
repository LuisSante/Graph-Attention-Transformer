import numpy as np
import config
from  utils.utils import init_weights, leaky_relu, softmax_with_mask
from data.dataset import create_graph, adj_to_edge_index, edge_index_to_adj
from models.gat_layer import GATLayer


if __name__ == "__main__":
    adj, features, labels = create_graph(num_nodes=6, num_features=5, num_classes=3, self_loops=True)
    print("Adjacency matrix:\n", adj)
    print("Features shape:", features.shape)
    print("Labels:", labels)

    gat_layer = GATLayer(in_features=features.shape[1], out_features=5, seed=config.SEED)
    h_out, alpha = gat_layer.forward(features, adj, return_attention=True)

    print("\nOutput features (h') shape:", h_out.shape)
    print(h_out)

    print("\nAttention coefficients (alpha) shape:", alpha.shape)
    print(alpha[:3, :3])

    row_sums = np.sum(alpha, axis=1)
    print("\nRow sums (deben ser 1.0 en nodos con vecinos):", row_sums)

