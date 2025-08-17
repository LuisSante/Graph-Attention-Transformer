import numpy as np
from models.gat_layer import MultiHeadGATLayer
from config import Config

class GATModel:
    def __init__(self, input_dim, hidden_dims, n_heads, 
                 dropout=0.0, alpha=0.2, seed=None, pooling_method='mean'):
        Config.print_separator(" INITIALIZING GAT MODEL FOR GRAPH CLASSIFICATION")
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if isinstance(hidden_dims, list) else [hidden_dims]
        self.n_heads = n_heads if isinstance(n_heads, list) else [n_heads]
        self.dropout = dropout
        self.alpha = alpha
        self.pooling_method = pooling_method  
        
        print(f" Model Configuration:")
        print(f"     Input dim: {input_dim}")
        print(f"     Hidden dims: {self.hidden_dims}")
        print(f"     N heads: {self.n_heads}")
        print(f"     Dropout: {dropout}")
        print(f"     Pooling method: {pooling_method}")  
        
        if len(self.n_heads) == 1:
            self.n_heads = self.n_heads * len(self.hidden_dims)

        assert len(self.n_heads) == len(self.hidden_dims), \
            "Number of heads must match number of hidden layers"
        
        self.layers = []
        current_input_dim = input_dim
        
        for i, (hidden_dim, heads) in enumerate(zip(self.hidden_dims, self.n_heads)):
            print(f"\n  Creating GAT layer {i+1}:") 
            print(f"    Input: {current_input_dim}, Output per head: {hidden_dim}, Heads: {heads}")
            
            layer_seed = None if seed is None else seed + i
            
            layer = MultiHeadGATLayer(
                in_features=current_input_dim,
                out_features_per_head=hidden_dim,
                n_heads=heads,
                concat=Config.CONCAT,
                seed=layer_seed
            )
            
            self.layers.append(layer)
            current_input_dim = heads * hidden_dim if Config.CONCAT else hidden_dim
                
        self.output_dim = current_input_dim
        print(f"    Final output dim: {self.output_dim}")
        
    
    def _elu_activation(self, x, alpha=1.0):
        print(f"  Applying ELU (α={alpha})")
        result = np.where(x > 0, x, alpha * (np.exp(x) - 1))
        print(f"       Min: {result.min():.4f}, Max: {result.max():.4f}")
        return result
    
    def _softmax_activation(self, x):
        print(f"  Applying Softmax")
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        result = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        print(f"       Sumas por fila: {np.sum(result, axis=1)}")
        return result
    
    def _graph_pooling(self, node_embeddings):
        Config.print_subsection(f" GRAPH POOLING - {self.pooling_method.upper()}")
        print(f"  Input node embeddings shape: {node_embeddings.shape}")
        
        if self.pooling_method == 'mean':
            graph_embedding = np.mean(node_embeddings, axis=0, keepdims=True)
            print(f"  Applied MEAN pooling")
        elif self.pooling_method == 'max':
            graph_embedding = np.max(node_embeddings, axis=0, keepdims=True)
            print(f"  Applied MAX pooling")
        elif self.pooling_method == 'sum':
            graph_embedding = np.sum(node_embeddings, axis=0, keepdims=True)
            print(f"  Applied SUM pooling")
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
        
        print(f"  Output graph embedding shape: {graph_embedding.shape}")
        print(f"  Graph embedding: {graph_embedding}")
        
        return graph_embedding

    def forward(self, features, adj, training=True, return_attention=True):
        Config.print_separator(" FORWARD PASS - GRAPH ENCODING MODEL")
        
        h = features
        all_attentions = []
        
        print(f"  INITIAL INPUT:")
        print(f"     Features shape: {features.shape}")
        print(f"     Adjacency shape: {adj.shape}")
        print(f"     Features:\n{features}")
        
        for i, layer in enumerate(self.layers):
            print(f"\n{'🟦'} PROCESSING GAT LAYER {i+1}/{len(self.layers)}") 
            print(f" Input shape: {h.shape}")
            
            if return_attention:
                h, layer_attentions = layer.forward(h, adj, return_attentions=True, training=training)
                all_attentions.append(layer_attentions)
            else:
                h = layer.forward(h, adj, return_attentions=False, training=training)
            
            print(f"   Output shape después de GAT: {h.shape}")
            
            if i < len(self.layers) - 1:
                print(f" Applying ELU activation...")
                h = self._elu_activation(h)
                print(f" Output shape after ELU: {h.shape}")
        
        final_graph_embedding = self._graph_pooling(h)
        final_graph_embedding = final_graph_embedding.flatten()  # (embedding_dim,)
        
        print(f"\n  FINAL GRAPH EMBEDDING (STOPS HERE):") 
        print(f"Shape: {final_graph_embedding.shape}") 
        print(f" Embedding: {final_graph_embedding}")
        print(f" L2 norm: {np.linalg.norm(final_graph_embedding):.4f}")
        print(f"\n This embedding is ready for:")
        print(f"   • MLP projection (if needed)")
        print(f"   • Graph similarity computation")
        print(f"   • Soft prompting for LLMs")
        
        if return_attention:
            return final_graph_embedding, all_attentions
        return final_graph_embedding