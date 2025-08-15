import numpy as np
from config import Config
from plots.graph import draw_graph_ascii, draw_graph_matplotlib

# Create a graph with 4 nodes (adjaceny matrix, edge list, self-loops)
def create_graph(num_nodes=4, num_features=5, num_classes=3, self_loops=False, seed=Config.SEED, visualize=True, show_features=True):
    Config.print_subsection("CREANDO GRAFO DE ENTRADA")
    rng = np.random.RandomState(seed)

    print(f"  Generating adjacency matrix {num_nodes}x{num_nodes}...")
    p = 0.3
    adj_upper = rng.rand(num_nodes, num_nodes) < p
    adj = np.triu(adj_upper, k=1)  
    adj = adj + adj.T              
    adj = adj.astype(np.float32)

    if self_loops:
        np.fill_diagonal(adj, 1.0)
        print("    Added self-loops to adjacency matrix.")

    print(f"    Generating node features {num_nodes}x{num_features}...") 
    features = rng.randn(num_nodes, num_features).astype(np.float32) 

    print(f"    Generating labels for {num_classes} classes...")
    labels = rng.randint(0, num_classes, size=num_nodes, dtype=np.int64)

    print(f"\n  GRAPH SUMMARY:") 
    print(f"    Nodes: {num_nodes}") 
    print(f"    Edges: {int(np.sum(adj) // 2)}") 
    print(f"    Features per node: {num_features}") 
    print(f"    Classes: {num_classes}") 
    print(f"    Self-loops: {'Yes' if self_loops else 'No'}")

    print(f"\n  ADJACENT MATRIX:") 
    print(f"{adj}") 

    print(f"\n  NODE CHARACTERISTICS:") 
    print(f"{features}") 

    print(f"\n  TAGS:") 
    print(f"{labels}")  

    if visualize:
        draw_graph_ascii(adj, labels, features=features)
        draw_graph_matplotlib(adj, labels, features=features, 
                            title=f"Graph with {num_nodes} nodes", 
                            show_features=show_features)

    return adj, features, labels

def add_self_loops(adj):
    adj_with_loops = adj.copy()
    np.fill_diagonal(adj_with_loops, 1.0)
    return adj_with_loops


def adj_to_edge_index(adj):
    src, dst = np.nonzero(adj)
    return np.vstack((src, dst))


def edge_index_to_adj(edge_index, num_nodes):
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    src, dst = edge_index
    adj[src, dst] = 1.0
    return adj