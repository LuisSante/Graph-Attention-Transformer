import numpy as np
from config import Config
from plots.graph import draw_graph_ascii, draw_graph_matplotlib

# Create a graph with 4 nodes (adjaceny matrix, edge list, self-loops)
def create_graph(num_nodes=4, num_features=5, num_classes=3, self_loops=False, seed=Config.SEED, 
                 directed=True, density=0.3, visualize=True, show_features=True):
    Config.print_subsection("CREANDO GRAFO DE ENTRADA")
    rng = np.random.RandomState(seed)

    print(f"  Generating {'directed' if directed else 'undirected'} adjacency matrix {num_nodes}x{num_nodes}...")
    
    if directed:
        adj = (rng.rand(num_nodes, num_nodes) < density).astype(np.float32)
        np.fill_diagonal(adj, 0.0)
        print(f"    Created directed graph with density ~{density}")
    else:
        adj_upper = rng.rand(num_nodes, num_nodes) < density
        adj = np.triu(adj_upper, k=1)  
        adj = adj + adj.T              
        adj = adj.astype(np.float32)
        print(f"    Created undirected graph with density ~{density}")

    if self_loops:
        np.fill_diagonal(adj, 1.0)
        print("    Added self-loops to adjacency matrix.")

    print(f"    Generating node features {num_nodes}x{num_features}...") 
    features = rng.randn(num_nodes, num_features).astype(np.float32) 

    print(f"    Generating labels for {num_classes} classes...")
    labels = rng.randint(0, num_classes, size=num_nodes, dtype=np.int64)

    total_edges = int(np.sum(adj))
    possible_edges = num_nodes * (num_nodes - 1) if not self_loops else num_nodes * num_nodes
    actual_density = total_edges / possible_edges

    print(f"\n  GRAPH SUMMARY:") 
    print(f"    Type: {'Directed' if directed else 'Undirected'}")
    print(f"    Nodes: {num_nodes}") 
    print(f"    Edges: {total_edges}") 
    print(f"    Density: {actual_density:.3f}")
    print(f"    Features per node: {num_features}") 
    print(f"    Classes: {num_classes}") 
    print(f"    Self-loops: {'Yes' if self_loops else 'No'}")

    print(f"\n  ADJACENCY MATRIX:") 
    print(f"{adj}") 

    print(f"\n  NODE FEATURES:") 
    print(f"{features}") 

    print(f"\n  LABELS:") 
    print(f"{labels}")  

    if visualize:
        draw_graph_ascii(adj, labels, features=features, directed=directed)
        draw_graph_matplotlib(adj, labels, features=features, show_features=show_features, directed=directed)

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