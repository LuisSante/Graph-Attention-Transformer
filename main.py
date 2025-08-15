import numpy as np
from config import Config
from data.dataset import create_graph
from utils.utils import get_graph_statistics
from models.gat_layer import MultiHeadGATLayer
from models.gat_model import GATModel

def run_complete_simulation():
    Config.print_separator("COMPLETE SIMULATION OF GRAPH ATTENTION NETWORK", "=", 100)
    print("     Objectives:")
    print("         ✓ Create a sample graph")
    print("         ✓ Initialize a GAT model")
    print("         ✓ Show the entire forward pass process")
    print("         ✓ Make predictions")
    print("         ✓ Analyze attention coefficients")
    
    # Crear datos de entrada
    adj, features, labels = create_graph(
        num_nodes=6, 
        num_features=Config.IN_FEATURES, 
        num_classes=Config.N_CLASSES, 
        self_loops=True, 
        seed=Config.SEED
    )
    
    # Mostrar estadísticas del grafo
    stats = get_graph_statistics(adj, features)
    Config.print_subsection(" GRAPH STATISTICS")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Crear modelo GAT
    Config.print_separator(" CREATING GAT MODEL")
    model = GATModel(
        input_dim=Config.IN_FEATURES,
        hidden_dims=[Config.HIDDEN_PER_HEAD],
        output_dim=Config.N_CLASSES,
        n_heads=[Config.N_HEADS],  # 3 cabezas para oculta, 1 para salida
        dropout=0.0,  # Sin dropout para simplificar
        seed=Config.SEED,
        final_activation='softmax'
    )
    
    # Forward pass completo con atención
    Config.print_separator(" FORWARD PASS WITH ATTENTION ANALYSIS")
    output, attentions = model.forward(features, adj, training=False, return_attention=True)
    
    # Análisis de atención
    Config.print_separator("ATTENTION QUOTIENT ANALYSIS")
    for layer_idx, layer_attentions in enumerate(attentions):
        Config.print_subsection(f"Layer {layer_idx + 1}")
        for head_idx, attention in enumerate(layer_attentions): 
            print(f"\n Head {head_idx + 1}:") 
            print(f"Shape: {attention.shape}") 
            print(f" Attention matrix:\n{attention}")
            
            # Encontrar las conexiones más fuertes
            max_attention = np.max(attention)
            max_pos = np.unravel_index(np.argmax(attention), attention.shape)
            print(f" Maximum attention: {max_attention:.4f} (node {max_pos[0]} → node {max_pos[1]})")
    
    # Realizar predicciones
    Config.print_separator(" PREDICCIONES FINALES")
    predictions = model.predict(features, adj)
    
    print(f"    COMPARISON PREDICTIONS vs. ACTUAL LABELS:")
    print(f"        Predictions: {predictions}")
    print(f"        Labels: {labels}")
    print(f"        Matches: {np.sum(predictions == labels)}/{len(labels)}")
    print(f"        Accuracy: {np.mean(predictions == labels)*100:.1f}%")
    
    # Resumen final
    Config.print_separator(" FINAL SUMMARY", "=", 100)
    print(f"    Simulation completed successfully")
    print(f"    Processed graph: {adj.shape[0]} nodes, {int(np.sum(adj)//2)} edges")
    print(f"    GAT model: {len(model.layers)} layers, {sum(model.n_heads)} total heads")
    print(f"    Predictions made with {np.mean(predictions == labels)*100:.1f}% accuracy")
    print(f"    The model assigned different attention weights to different neighbors")
    print(f"    Transformation process: {Config.IN_FEATURES} → {Config.HIDDEN_PER_HEAD * Config.N_HEADS} → {Config.N_CLASSES}")

if __name__ == "__main__":
    # Configurar numpy para mejor visualización
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    
    # Ejecutar simulación
    run_complete_simulation()