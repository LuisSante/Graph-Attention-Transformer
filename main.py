import numpy as np
from config import Config
from data.dataset import create_graph, create_multiple_graphs, pad_graphs_batch
from utils.utils import get_graph_statistics, get_batch_statistics
from models.gat_model import GATModel

def run_single_graph_simulation():
    Config.print_separator("SINGLE GRAPH SIMULATION - GRAPH ENCODING", "=", 100)
    
    adj, features = create_graph(
        num_nodes=Config.N_NODES, 
        num_features=Config.IN_FEATURES, 
        directed=Config.DIRECTED,  
        density=Config.GRAPH_DENSITY, 
        seed=Config.SEED
    )
    
    stats = get_graph_statistics(adj, features, directed=Config.DIRECTED)
    Config.print_subsection(" GRAPH STATISTICS")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    Config.print_separator(" CREATING GAT MODEL FOR GRAPH ENCODING")
    model = GATModel(
        input_dim=Config.IN_FEATURES,
        hidden_dims=[Config.OUT_FEATURES_PER_HEAD], 
        n_heads=[Config.N_HEADS],
        dropout=0.0,
        seed=Config.SEED,
        pooling_method='mean'
    )
    
    Config.print_separator(" FORWARD PASS - SINGLE GRAPH ENCODING")
    graph_embedding, attentions = model.forward(features, adj, training=False, return_attention=True)
    
    return graph_embedding, attentions

def run_multi_graph_simulation():
    Config.print_separator("MULTI-GRAPH SIMULATION - BATCH GRAPH ENCODING", "=", 100)
    
    graphs_data = create_multiple_graphs(
        n_graphs=Config.N_GRAPHS,
        min_nodes=4,
        max_nodes=Config.N_NODES,
        num_features=Config.IN_FEATURES,
        directed=Config.DIRECTED,
        density=Config.GRAPH_DENSITY,
        seed=Config.SEED
    )
    
    batch_stats = get_batch_statistics(graphs_data, directed=Config.DIRECTED)
    Config.print_subsection(" BATCH STATISTICS")
    for key, value in batch_stats.items():
        print(f"  {key}: {value}")
    
    Config.print_separator(" CREATING GAT MODEL FOR BATCH PROCESSING")
    model = GATModel(
        input_dim=Config.IN_FEATURES,
        hidden_dims=[Config.OUT_FEATURES_PER_HEAD], 
        n_heads=[Config.N_HEADS],
        dropout=0.0,
        seed=Config.SEED,
        pooling_method='mean',
        batch_processing=Config.BATCH_PROCESSING
    )
    
    if Config.BATCH_PROCESSING == "padding":
        Config.print_separator(" PADDING GRAPHS FOR BATCH PROCESSING")
        padded_adj, padded_features, masks = pad_graphs_batch(
            graphs_data, max_nodes=Config.MAX_NODES
        )
        
        Config.print_separator(" FORWARD PASS - PADDED BATCH PROCESSING")
        batch_embeddings, batch_attentions = model.forward_batch_padded(
            padded_features, padded_adj, masks, 
            training=False, return_attention=True
        )
        
    else:  # sequential processing
        Config.print_separator(" FORWARD PASS - SEQUENTIAL BATCH PROCESSING")
        batch_embeddings, batch_attentions = model.forward_batch_sequential(
            graphs_data, training=False, return_attention=True
        )
    
    return batch_embeddings, batch_attentions

def run_complete_simulation():
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    
    if Config.DETAIL_GRAPH == "single_graph":
        return run_single_graph_simulation()
    elif Config.DETAIL_GRAPH == "multi_graph":
        return run_multi_graph_simulation()
    else:
        raise ValueError(f"Invalid DETAIL_GRAPH setting: {Config.DETAIL_GRAPH}")

if __name__ == "__main__":
    embeddings, attentions = run_complete_simulation()
