import numpy as np
from config import Config
from data.dataset import create_graph
from utils.utils import get_graph_statistics
from models.gat_model import GATModel

def run_complete_simulation():
    Config.print_separator("COMPLETE SIMULATION OF GRAPH ATTENTION NETWORK - GRAPH ENCODING", "=", 100)
    print("     Objectives:")
    print("         ✓ Create a sample directed graph") 
    print("         ✓ Initialize a GAT model for GRAPH representation learning")
    print("         ✓ Show the entire forward pass process")
    print("         ✓ Extract dense vector representation of the ENTIRE GRAPH")
    print("         ✓ Analyze asymmetric attention coefficients") 
    print("         ✗ NO classification - just encoding like G-Retriever")
    
    adj, features = create_graph(
        num_nodes=Config.N_NODES, 
        num_features=Config.IN_FEATURES, 
        directed=Config.DIRECTED,  
        density=Config.GRAPH_DENSITY, 
        seed=Config.SEED,
        visualize=True
    )
    
    stats = get_graph_statistics(adj, features, directed=Config.DIRECTED)
    Config.print_subsection(" GRAPH STATISTICS")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    Config.print_separator(" CREATING GAT MODEL FOR GRAPH ENCODING")
    model = GATModel(
        input_dim=Config.IN_FEATURES,
        hidden_dims=[Config.HIDDEN_PER_HEAD], 
        output_dim=Config.GRAPH_EMBEDDING_DIM, 
        n_heads=[Config.N_HEADS],
        dropout=0.0,
        seed=Config.SEED,
        pooling_method='max'  
    )
    
    Config.print_separator(" FORWARD PASS - GRAPH ENCODING")
    graph_embedding, attentions = model.forward(features, adj, training=False, return_attention=True)
    
    print(f"\n GRAPH ENCODING RESULTS:")
    print(f"   Graph embedding shape: {graph_embedding.shape}")
    print(f"   Graph embedding: {graph_embedding}")
    print(f"   L2 norm: {np.linalg.norm(graph_embedding):.4f}")
    print(f"\n   This embedding can now be used for:")
    print(f"     • Similarity search between graphs")
    print(f"     • Input to LLM via soft prompting")
    print(f"     • Downstream tasks like classification")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    run_complete_simulation()