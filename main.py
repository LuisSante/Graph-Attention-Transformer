import numpy as np
from config import Config
from data.dataset import create_graph, create_multiple_graphs, pad_graphs_batch, generate_graph_labels
from utils.utils import get_graph_statistics, get_batch_statistics
from models.gat_mlp import GatMLP

def run_single_graph_classification():
    Config.print_separator("SINGLE GRAPH CLASSIFICATION", "=", 100)
    
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
    
    gat_config = {
        'input_dim': Config.IN_FEATURES,
        'hidden_dims': [Config.OUT_FEATURES_PER_HEAD],
        'n_heads': [Config.N_HEADS],
        'dropout': 0.0,
        'pooling_method': 'mean'
    }
    
    mlp_config = {
        'hidden_dims': Config.MLP_HIDDEN_DIMS,
        'num_classes': Config.NUM_CLASSES,
        'dropout': Config.MLP_DROPOUT,
        'activation': Config.MLP_ACTIVATION
    }
    
    model = GatMLP(gat_config, mlp_config, seed=Config.SEED)
    
    results = model.forward(
        features, adj, 
        training=False,
        return_attention=Config.RETURN_ATTENTION,
        return_embeddings=Config.RETURN_EMBEDDINGS
    )
    
    return results

def run_multi_graph_classification():
    Config.print_separator("MULTI-GRAPH CLASSIFICATION", "=", 100)
    
    graphs_data = create_multiple_graphs(
        n_graphs=Config.N_GRAPHS,
        min_nodes=4,
        max_nodes=Config.N_NODES,
        num_features=Config.IN_FEATURES,
        directed=Config.DIRECTED,
        density=Config.GRAPH_DENSITY,
        seed=Config.SEED
    )
    
    labels = generate_graph_labels(
        graphs_data, 
        num_classes=Config.NUM_CLASSES,
        seed=Config.SEED,
        label_strategy=Config.LABEL_STRATEGY
    )
    
    batch_stats = get_batch_statistics(graphs_data, directed=Config.DIRECTED)
    Config.print_subsection(" BATCH STATISTICS")
    for key, value in batch_stats.items():
        print(f"  {key}: {value}")
    
    gat_config = {
        'input_dim': Config.IN_FEATURES,
        'hidden_dims': [Config.OUT_FEATURES_PER_HEAD],
        'n_heads': [Config.N_HEADS],
        'dropout': 0.0,
        'pooling_method': 'mean',
        'batch_processing': Config.BATCH_PROCESSING
    }
    
    mlp_config = {
        'hidden_dims': Config.MLP_HIDDEN_DIMS,
        'num_classes': Config.NUM_CLASSES,
        'dropout': Config.MLP_DROPOUT,
        'activation': Config.MLP_ACTIVATION
    }
    
    model = GatMLP(gat_config, mlp_config, seed=Config.SEED)
    
    if Config.BATCH_PROCESSING == "padding":
        Config.print_separator(" PADDING GRAPHS FOR BATCH PROCESSING")
        padded_adj, padded_features, masks = pad_graphs_batch(
            graphs_data, max_nodes=Config.MAX_NODES
        )
        
        batch_data = {
            'features': padded_features,
            'adj': padded_adj,
            'masks': masks
        }
    else:
        batch_data = graphs_data
    
    results = model.forward_batch(
        batch_data,
        training=False,
        return_attention=Config.RETURN_ATTENTION,
        return_embeddings=Config.RETURN_EMBEDDINGS
    )
    
    Config.print_separator(" CLASSIFICATION RESULTS")
    predicted_classes = results['predicted_classes']
    true_labels = labels
    
    print(f"  True labels:      {true_labels}")
    print(f"  Predicted labels: {predicted_classes}")
    
    accuracy = np.mean(predicted_classes == true_labels)
    print(f"  Accuracy: {accuracy:.2%}")
    
    return results

def run_complete_classification():
    np.set_printoptions(precision=4, suppress=True, linewidth=130)
    
    if Config.DETAIL_GRAPH == "single_graph":
        return run_single_graph_classification()
    elif Config.DETAIL_GRAPH == "multi_graph":
        return run_multi_graph_classification()
    else:
        raise ValueError(f"Invalid DETAIL_GRAPH setting: {Config.DETAIL_GRAPH}")

if __name__ == "__main__":
    results = run_complete_classification()