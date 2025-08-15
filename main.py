import numpy as np
from config import Config
from data.dataset import create_graph
from utils.utils import get_graph_statistics
from models.gat_model import GATModel

def run_complete_simulation():
    Config.print_separator("COMPLETE SIMULATION OF GRAPH ATTENTION NETWORK", "=", 100)
    print("     Objectives:")
    print("         ✓ Create a sample directed graph") 
    print("         ✓ Initialize a GAT model")
    print("         ✓ Show the entire forward pass process")
    print("         ✓ Make predictions")
    print("         ✓ Analyze asymmetric attention coefficients") 
    
    adj, features, labels = create_graph(
        num_nodes=Config.N_NODES, 
        num_features=Config.IN_FEATURES, 
        num_classes=Config.N_CLASSES, 
        directed=Config.DIRECTED,  
        density=Config.GRAPH_DENSITY, 
        seed=Config.SEED,
        visualize=True
    )
    
    stats = get_graph_statistics(adj, features, directed=Config.DIRECTED)  # Actualizado
    Config.print_subsection(" GRAPH STATISTICS")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    Config.print_separator(" CREATING GAT MODEL")
    model = GATModel(
        input_dim=Config.IN_FEATURES,
        hidden_dims=[Config.HIDDEN_PER_HEAD], # Controlar el numero de capas
        output_dim=Config.N_CLASSES,
        n_heads=[Config.N_HEADS],  # 3 cabezas para oculta, 1 para salida
        dropout=0.0,  # Sin dropout para simplificar
        seed=Config.SEED,
        final_activation='softmax'
    )
    
    Config.print_separator(" FORWARD PASS WITH ATTENTION ANALYSIS")
    _, attentions = model.forward(features, adj, training=False, return_attention=True)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    run_complete_simulation()