import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.dataset import create_graph, add_self_loops
from models.gat_model import GATModel, create_transductive_gat, create_inductive_gat
from utils.utils import get_graph_statistics
import config

def test_basic_functionality():
    print("🧪 Testing Basic GAT Functionality")
    print("=" * 50)
    
    adj, features, labels = create_graph(num_nodes=6, num_features=5, num_classes=3, self_loops=True, seed=config.SEED)
    
    print(f"Graph: {adj.shape[0]} nodes, {features.shape[1]} features, {len(np.unique(labels))} classes")
    print(f"Graph statistics: {get_graph_statistics(adj, features)}")
    
    print("\n📍 Testing Single GAT Layer:")
    from models.gat_layer import GATLayer
    
    gat_layer = GATLayer(in_features=5, out_features=8, seed=config.SEED)
    output, attention = gat_layer.forward(features, adj, return_attention=True)
    
    print(f"Input shape: {features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attention.shape}")
    print(f"Attention row sums: {np.sum(attention, axis=1)[:3]}")  # Should be ~1.0
    
    print("\n📍 Testing Multi-Head GAT Layer:")
    from models.gat_layer import MultiHeadGATLayer
    
    multihead_gat = MultiHeadGATLayer(in_features=5, out_features_per_head=8, n_heads=3, 
                                     concat=True, seed=config.SEED)
    
    output, attentions = multihead_gat.forward(features, adj, return_attentions=True)
    
    print(f"Multi-head output shape: {output.shape}")
    print(f"Expected shape: (6, {3 * 8})")
    print(f"Number of attention matrices: {len(attentions)}")
    
    return True
