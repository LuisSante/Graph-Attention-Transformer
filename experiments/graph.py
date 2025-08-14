import numpy as np
from config import Config
from data.dataset import create_graph
from models.gat_layer import MultiHeadGATLayer

if __name__ == "__main__":
    adj, features, labels = create_graph(num_nodes=6, num_features=5, num_classes=3, self_loops=True)
    print("Adjacency matrix:\n", adj)
    print("Features shape:", features.shape)
    print("Labels:", labels)

    multihead_gat = MultiHeadGATLayer(in_features=features.shape[1], out_features_per_head=5, n_heads=3, concat=True, seed=Config.SEED)

    # Forward
    h_out, alphas = multihead_gat.forward(features, adj, return_attentions=True)

    print("\nOutput features (h') shape:", h_out.shape)
    print(h_out)

    for idx, alpha in enumerate(alphas):
        print(f"\nHead {idx} alpha (3x3 sample):\n", alpha[:3, :3])
        row_sums = np.sum(alpha, axis=1)
        print("Row sums (deben ser 1.0 en nodos con vecinos):", row_sums)

