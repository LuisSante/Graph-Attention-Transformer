import numpy as np
from models.gat_layer import MultiHeadGATLayer
from config import Config

class GATModel:
    def __init__(self, input_dim, hidden_dims, output_dim, n_heads, 
                 dropout=0.0, alpha=0.2, seed=None, final_activation='softmax'):
        Config.print_separator(" INITIALIZING COMPLETE GAT MODEL")
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if isinstance(hidden_dims, list) else [hidden_dims]
        self.output_dim = output_dim
        self.n_heads = n_heads if isinstance(n_heads, list) else [n_heads]
        self.dropout = dropout
        self.alpha = alpha
        self.final_activation = final_activation
        
        print(f" Model Configuration:")
        print(f"     Input dim: {input_dim}")
        print(f"     Hidden dims: {self.hidden_dims}")
        print(f"     Output dim: {output_dim}")
        print(f"     N heads: {self.n_heads}")
        print(f"     Dropout: {dropout}")
        print(f"     Final activation: {final_activation}")
        
        # Ajustar número de cabezas si es necesario
        if len(self.n_heads) == 1:
            self.n_heads = self.n_heads * len(self.hidden_dims)

        assert len(self.n_heads) == len(self.hidden_dims), \
            "Number of heads must match number of hidden layers"
        
        # Crear capas ocultas
        self.layers = []
        current_input_dim = input_dim
        
        for i, (hidden_dim, heads) in enumerate(zip(self.hidden_dims, self.n_heads)):
            print(f"\n Creating hidden layer {i+1}:") 
            print(f" Input: {current_input_dim}, Output per head: {hidden_dim}, Heads: {heads}")
            
            layer_seed = None if seed is None else seed + i
            
            layer = MultiHeadGATLayer(
                in_features=current_input_dim,
                out_features_per_head=hidden_dim,
                n_heads=heads,
                concat=Config.CONCAT,
                seed=layer_seed
            )
            
            self.layers.append(layer)
            current_input_dim = heads * hidden_dim
        
        # Crear capa de salida
        print(f"\n  Creating output layer:")
        output_seed = None if seed is None else seed + len(self.hidden_dims)
        output_heads = 1
        
        if self.final_activation == 'sigmoid':
            output_heads = 6
        
        print(f"     Input: {current_input_dim}, Output: {output_dim}, Heads: {output_heads}")
        
        self.output_layer = MultiHeadGATLayer(
            in_features=current_input_dim,
            out_features_per_head=output_dim,
            n_heads=output_heads,
            concat=False,  # Promedio para capa de salida
            seed=output_seed
        )
        
        self.layers.append(self.output_layer)
        
        print(f"\n  GAT model initialized with {len(self.layers)} layers")
    
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
    
    def _sigmoid_activation(self, x):
        print(f"  Applying Sigmoid")
        result = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        print(f"       Min: {result.min():.4f}, Max: {result.max():.4f}")
        return result

    def forward(self, features, adj, training=True, return_attention=True):
        Config. print_separator(" FORWARD PASS - MODELO COMPLETO")
        
        h = features
        all_attentions = []
        
        print(f"  INITIAL INPUT:")
        print(f"     Features shape: {features.shape}")
        print(f"     Adjacency shape: {adj.shape}")
        print(f"     Features:\n{features}")
        
        for i, layer in enumerate(self.layers):
            print(f"\n{'🟦' if i < len(self.layers)-1 else '🟩'} PROCESSING LAYER {i+1}/{len(self.layers)} {'(HIDE)' if i < len(self.layers)-1 else '(OUTPUT)'}") 
            print(f" Input shape: {h.shape}")
            
            if return_attention:
                h, layer_attentions = layer.forward(h, adj, return_attentions=True, training=training)
                all_attentions.append(layer_attentions)
            else:
                h = layer.forward(h, adj, return_attentions=False, training=training)
            
            print(f"   Output shape después de GAT: {h.shape}")
            
            # Apply activation after each hidden layer
            if i < len(self.layers) - 1:
                print(f" Applying ELU activation...")
                h = self._elu_activation(h)
                print(f" Output shape after ELU: {h.shape}")
        
        # Aplicar activación final
        Config.print_subsection(" APPLYING FINAL ACTIVATION")
        if self.final_activation == 'softmax':
            h = self._softmax_activation(h)
        elif self.final_activation == 'sigmoid':
            h = self._sigmoid_activation(h)
        
        print(f"\n FINAL MODEL OUTPUT:") 
        print(f"Shape: {h.shape}") 
        print(f" Values:\n{h}")
        
        if return_attention:
            return h, all_attentions
        return h
