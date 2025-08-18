import numpy as np
from config import Config

def create_graph(num_nodes=4, num_features=5, self_loops=False, seed=Config.SEED, 
                 directed=True, density=0.3):
    print("CREATING INPUT GRAPH - GRAPH ENCODING")
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

    total_edges = int(np.sum(adj))
    possible_edges = num_nodes * (num_nodes - 1) if not self_loops else num_nodes * num_nodes
    actual_density = total_edges / possible_edges

    print(f"\n  GRAPH SUMMARY:") 
    print(f"    Type: {'Directed' if directed else 'Undirected'}")
    print(f"    Nodes: {num_nodes}") 
    print(f"    Edges: {total_edges}") 
    print(f"    Density: {actual_density:.3f}")
    print(f"    Features per node: {num_features}") 
    print(f"    Self-loops: {'Yes' if self_loops else 'No'}")
    print(f"    Target: Graph-level embedding (NO labels)")

    print(f"\n  ADJACENCY MATRIX:") 
    print(f"{adj}") 

    print(f"\n  NODE FEATURES (unlabeled):") 
    print(f"{features}") 

    return adj, features 

def create_multiple_graphs(n_graphs, min_nodes=3, max_nodes=8, num_features=5, 
                          directed=True, density=0.3, seed=42):
    Config.print_subsection("CREATING MULTIPLE GRAPHS BATCH")
    rng = np.random.RandomState(seed)
    graphs_data = []
    
    print(f"  Creating {n_graphs} graphs with {min_nodes}-{max_nodes} nodes each...")
    
    for i in range(n_graphs):
        num_nodes = rng.randint(min_nodes, max_nodes + 1)
        graph_seed = seed + i * 100
        
        Config.print_subsection(f"GRAPH {i+1}/{n_graphs} - {num_nodes} nodes")
        
        adj, features = create_graph(
            num_nodes=num_nodes,
            num_features=num_features,
            directed=directed,
            density=density,
            seed=graph_seed
        )
        
        graphs_data.append({
            'adj': adj,
            'features': features,
            'num_nodes': num_nodes,
            'graph_id': i
        })
        
        print(f"      Shape: adj{adj.shape}, features{features.shape}")
        print(f"      Edges: {int(np.sum(adj))}")
    
    print(f"\n   Created batch of {n_graphs} graphs")
    return graphs_data

def pad_graphs_batch(graphs_data, max_nodes=None):
    Config.print_subsection("PADDING GRAPHS TO BATCH SIZE")
    
    if max_nodes is None:
        max_nodes = max(graph['num_nodes'] for graph in graphs_data)
    
    n_graphs = len(graphs_data)
    n_features = graphs_data[0]['features'].shape[1]
    
    print(f"  Padding {n_graphs} graphs to {max_nodes} nodes each...")
    
    batch_adj = np.zeros((n_graphs, max_nodes, max_nodes), dtype=np.float32)
    batch_features = np.zeros((n_graphs, max_nodes, n_features), dtype=np.float32)
    batch_masks = np.zeros((n_graphs, max_nodes), dtype=bool)
    
    for i, graph in enumerate(graphs_data):
        num_nodes = graph['num_nodes']
        adj = graph['adj']
        features = graph['features']
        
        batch_adj[i, :num_nodes, :num_nodes] = adj
        
        batch_features[i, :num_nodes, :] = features
        
        batch_masks[i, :num_nodes] = True
        
        print(f"    Graph {i+1}: {num_nodes} nodes -> padded to {max_nodes}")
        print(f"      Original edges: {int(np.sum(adj))}")
        print(f"      Padded edges: {int(np.sum(batch_adj[i]))}")
    
    print(f"\n   BATCH SHAPES:")
    print(f"    Adjacency: {batch_adj.shape}")
    print(f"    Features: {batch_features.shape}")
    print(f"    Masks: {batch_masks.shape}")
    print(f"    Real nodes per graph: {batch_masks.sum(axis=1)}")
    
    return batch_adj, batch_features, batch_masks
