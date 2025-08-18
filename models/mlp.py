import numpy as np
from utils.utils import init_weights
from config import Config

class MLP:
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.0, activation='relu', seed=None):
        Config.print_title(" INITIALIZING MLP CLASSIFIER")
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if isinstance(hidden_dims, list) else [hidden_dims]
        self.num_classes = num_classes
        self.dropout = dropout
        self.activation = activation
        self.seed = seed
        
        print(f" MLP Configuration:")
        print(f"     Input dim: {input_dim}")
        print(f"     Hidden dims: {self.hidden_dims}")
        print(f"     Output classes: {num_classes}")
        print(f"     Dropout: {dropout}")
        print(f"     Activation: {activation}")
        
        # Initialize layers
        self.layers = []
        layer_dims = [input_dim] + self.hidden_dims + [num_classes]
        
        for i in range(len(layer_dims) - 1):
            layer_seed = None if seed is None else seed + i * 10
            weight = init_weights(layer_dims[i], layer_dims[i + 1], layer_seed)
            bias = np.zeros(layer_dims[i + 1], dtype=np.float32)
            
            self.layers.append({
                'weight': weight,
                'bias': bias,
                'input_dim': layer_dims[i],
                'output_dim': layer_dims[i + 1]
            })
            
            print(f"  Layer {i+1}: {layer_dims[i]} -> {layer_dims[i + 1]}")
        
        print(f"  MLP initialized with {len(self.layers)} layers")
    
    def _apply_activation(self, x, layer_idx):
        if self.activation == 'relu':
            result = np.maximum(0, x)
            print(f"      ReLU activation applied")
        elif self.activation == 'elu':
            result = np.where(x > 0, x, np.exp(x) - 1)
            print(f"      ELU activation applied")
        elif self.activation == 'tanh':
            result = np.tanh(x)
            print(f"      Tanh activation applied")
        else:
            result = x
            print(f"      No activation applied")
        
        print(f"      Output range: [{result.min():.4f}, {result.max():.4f}]")
        return result
    
    def _apply_dropout(self, x, training=True):
        if training and self.dropout > 0:
            rng = np.random.RandomState(self.seed)
            mask = rng.binomial(1, 1 - self.dropout, x.shape) / (1 - self.dropout)
            result = x * mask
            print(f"      Dropout applied (rate={self.dropout})")
            return result
        return x
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, graph_embeddings, training=True, return_logits=False):
        Config.print_separator(" MLP FORWARD PASS")
        
        single_graph = False
        if graph_embeddings.ndim == 1:
            single_graph = True
            x = graph_embeddings.reshape(1, -1)
            print(f"  Single graph input reshaped: {graph_embeddings.shape} -> {x.shape}")
        else:
            x = graph_embeddings
            print(f"  Batch input shape: {x.shape}")
        
        batch_size = x.shape[0]
        print(f"  Processing {batch_size} graph(s)")
        
        for i, layer in enumerate(self.layers):
            print(f"\n  MLP Layer {i+1}/{len(self.layers)}:")
            print(f"    Input shape: {x.shape}")
            
            x = np.dot(x, layer['weight']) + layer['bias']
            print(f"    After linear: {x.shape}")
            
            if i < len(self.layers) - 1:
                x = self._apply_activation(x, i)
                x = self._apply_dropout(x, training)
            else:
                print(f"    Output layer - no activation")
                print(f"    Logits range: [{x.min():.4f}, {x.max():.4f}]")
        
        if not return_logits:
            print(f"\n  Applying softmax...")
            x = self._softmax(x)
            print(f"  Probabilities shape: {x.shape}")
            print(f"  Sample probabilities sum: {x[0].sum():.6f}")
        
        if single_graph:
            x = x.squeeze(0)
            print(f"  Single graph output shape: {x.shape}")
        
        return x
    
    def predict(self, graph_embeddings):
        probabilities = self.forward(graph_embeddings, training=False)
        if probabilities.ndim == 1:
            return np.argmax(probabilities)
        else:
            return np.argmax(probabilities, axis=1)