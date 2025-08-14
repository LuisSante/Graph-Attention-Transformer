import numpy as np
from models.gat_layer import MultiHeadGATLayer
from utils.utils import init_weights


class GATModel:
    def __init__(self, input_dim, hidden_dims, output_dim, n_heads, 
                 dropout=0.0, alpha=0.2, seed=None, final_activation='softmax'):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if isinstance(hidden_dims, list) else [hidden_dims]
        self.output_dim = output_dim
        self.n_heads = n_heads if isinstance(n_heads, list) else [n_heads]
        self.dropout = dropout
        self.alpha = alpha
        self.final_activation = final_activation
        
        if len(self.n_heads) == 1:
            self.n_heads = self.n_heads * len(self.hidden_dims)
        
        assert len(self.n_heads) == len(self.hidden_dims), \
            "Number of heads must match number of hidden layers"
        
        self.layers = []
        current_input_dim = input_dim
        
        for i, (hidden_dim, heads) in enumerate(zip(self.hidden_dims, self.n_heads)):
            layer_seed = None if seed is None else seed + i
            
            layer = MultiHeadGATLayer(
                in_features=current_input_dim,
                out_features_per_head=hidden_dim,
                n_heads=heads,
                concat=True,  # Concatenate hidden layers
                dropout=dropout,
                seed=layer_seed
            )
            
            self.layers.append(layer)
            current_input_dim = heads * hidden_dim  # Concatenated output size
        
        output_seed = None if seed is None else seed + len(self.hidden_dims)
        output_heads = 1  # Usually single head for output, or multiple for averaging
        
        if self.final_activation == 'sigmoid':  # Multi-label case
            output_heads = 6  # As used in PPI dataset in the paper
        
        self.output_layer = MultiHeadGATLayer(
            in_features=current_input_dim,
            out_features_per_head=output_dim,
            n_heads=output_heads,
            concat=False,  # Average for output layer
            dropout=0.0,   # No dropout on output layer
            seed=output_seed
        )
        
        self.layers.append(self.output_layer)
    
    def forward(self, features, adj, training=True, return_attention=False):
        h = features
        all_attentions = []
        
        for i, layer in enumerate(self.layers):
            if return_attention:
                h, layer_attentions = layer.forward(h, adj, return_attentions=True, training=training)
                all_attentions.append(layer_attentions)
            else:
                h = layer.forward(h, adj, return_attentions=False, training=training)
            
            # Apply activation after each hidden layer (except output)
            if i < len(self.layers) - 1:
                h = self._elu_activation(h)
        
        # Apply final activation
        if self.final_activation == 'softmax':
            h = self._softmax_activation(h)
        elif self.final_activation == 'sigmoid':
            h = self._sigmoid_activation(h)
        
        if return_attention:
            return h, all_attentions
        return h
    
    def predict(self, features, adj):
        logits = self.forward(features, adj, training=False)
        
        if self.final_activation == 'softmax':
            # Multi-class: return class with highest probability
            return np.argmax(logits, axis=1)
        elif self.final_activation == 'sigmoid':
            # Multi-label: return binary predictions
            return (logits > 0.5).astype(int)
        else:
            return logits
    
    def get_embeddings(self, features, adj, layer_idx=-2):
        h = features
        
        target_layer = len(self.layers) + layer_idx if layer_idx < 0 else layer_idx
        
        for i, layer in enumerate(self.layers):
            h = layer.forward(h, adj, training=False)
            
            if i == target_layer:
                return h
            
            if i < len(self.layers) - 1:
                h = self._elu_activation(h)
        
        return h
    
    def get_attention_weights(self, features, adj):
        _, all_attentions = self.forward(features, adj, training=False, return_attention=True)
        return all_attentions
    
    def _elu_activation(self, x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def _softmax_activation(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _sigmoid_activation(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    def get_model_info(self):
        total_params = 0
        layer_info = []
        
        for i, layer in enumerate(self.layers):
            layer_params = 0
            for head in layer.heads:
                layer_params += head.W.shape[0] * head.W.shape[1]
                layer_params += len(head.a)
            
            total_params += layer_params
            
            layer_info.append({
                'layer': i,
                'type': 'MultiHeadGAT',
                'input_dim': layer.heads[0].in_features,
                'output_dim': layer.output_dim,
                'n_heads': layer.n_heads,
                'parameters': layer_params
            })
        
        return {
            'total_parameters': total_params,
            'layers': layer_info,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim
        }


def create_transductive_gat(input_dim, output_dim, seed=None):
    return GATModel(
        input_dim=input_dim,
        hidden_dims=[8],  # 8 features per head
        output_dim=output_dim,
        n_heads=[8, 1],   # 8 heads for hidden, 1 for output
        dropout=0.6,
        seed=seed,
        final_activation='softmax'
    )


def create_inductive_gat(input_dim, output_dim, seed=None):
    return GATModel(
        input_dim=input_dim,
        hidden_dims=[256, 256],  # Two hidden layers with 256 features per head
        output_dim=output_dim,
        n_heads=[4, 4, 6],       # 4 heads for hidden layers, 6 for output
        dropout=0.0,             # No dropout for inductive
        seed=seed,
        final_activation='sigmoid'  # Multi-label classification
    )