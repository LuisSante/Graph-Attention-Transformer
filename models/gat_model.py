import numpy as np
from models.gat_layer import MultiHeadGATLayer
from config import Config

class GATModel:
    def __init__(self, input_dim, hidden_dims, n_heads, 
                 dropout=0.0, alpha=0.2, seed=None, pooling_method='mean', 
                 batch_processing='padding'):
        Config.print_separator(" INITIALIZING GAT MODEL")
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if isinstance(hidden_dims, list) else [hidden_dims]
        self.n_heads = n_heads if isinstance(n_heads, list) else [n_heads]
        self.dropout = dropout
        self.alpha = alpha
        self.pooling_method = pooling_method
        self.batch_processing = batch_processing
        
        print(f" Model Configuration:")
        print(f"     Input dim: {input_dim}")
        print(f"     Hidden dims: {self.hidden_dims}")
        print(f"     N heads: {self.n_heads}")
        print(f"     Dropout: {dropout}")
        print(f"     Pooling method: {pooling_method}")
        print(f"     Batch processing: {batch_processing}")
        
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
    
    def _graph_pooling(self, node_embeddings, mask=None):
        Config.print_subsection(f" GRAPH POOLING - {self.pooling_method.upper()}")
        print(f"  Input node embeddings shape: {node_embeddings.shape}")
        
        if mask is not None:
            print(f"  Using mask: {mask.sum()} valid nodes out of {len(mask)}")
            masked_embeddings = node_embeddings * mask.reshape(-1, 1)
            valid_count = mask.sum()
            
            if self.pooling_method == 'mean':
                graph_embedding = np.sum(masked_embeddings, axis=0) / max(valid_count, 1)
                print(f"  Applied MASKED MEAN pooling")
            elif self.pooling_method == 'max':
                masked_embeddings = np.where(mask.reshape(-1, 1), masked_embeddings, -np.inf)
                graph_embedding = np.max(masked_embeddings, axis=0)
                print(f"  Applied MASKED MAX pooling")
            elif self.pooling_method == 'sum':
                graph_embedding = np.sum(masked_embeddings, axis=0)
                print(f"  Applied MASKED SUM pooling")
        else:
            if self.pooling_method == 'mean':
                graph_embedding = np.mean(node_embeddings, axis=0)
                print(f"  Applied MEAN pooling")
            elif self.pooling_method == 'max':
                graph_embedding = np.max(node_embeddings, axis=0)
                print(f"  Applied MAX pooling")
            elif self.pooling_method == 'sum':
                graph_embedding = np.sum(node_embeddings, axis=0)
                print(f"  Applied SUM pooling")
        
        print(f"  Output graph embedding shape: {graph_embedding.shape}")
        print(f"  Graph embedding: {graph_embedding}")
        
        return graph_embedding

    def forward(self, features, adj, training=True, return_attention=True):
        Config.print_separator(" FORWARD PASS - SINGLE GRAPH ENCODING")
        
        h = features
        all_attentions = []
        
        print(f"  INITIAL INPUT:")
        print(f"     Features shape: {features.shape}")
        print(f"     Adjacency shape: {adj.shape}")
        print(f"     Features:\n{features}")
        
        for i, layer in enumerate(self.layers):
            print(f"\n PROCESSING GAT LAYER {i+1}/{len(self.layers)}") 
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
        
        print(f"\n  FINAL GRAPH EMBEDDING:") 
        print(f"Shape: {final_graph_embedding.shape}") 
        print(f" Embedding: {final_graph_embedding}")
        print(f" L2 norm: {np.linalg.norm(final_graph_embedding):.4f}")
        
        if return_attention:
            return final_graph_embedding, all_attentions
        return final_graph_embedding

    def forward_batch_padded(self, batch_features, batch_adj, batch_masks, 
                           training=True, return_attention=True):
        Config.print_separator(" FORWARD PASS - PADDED BATCH PROCESSING")
        
        n_graphs = batch_features.shape[0]
        batch_embeddings = []
        batch_all_attentions = []
        
        print(f"  BATCH INPUT:")
        print(f"     Batch size: {n_graphs}")
        print(f"     Padded features shape: {batch_features.shape}")
        print(f"     Padded adjacency shape: {batch_adj.shape}")
        print(f"     Masks shape: {batch_masks.shape}")
        
        for graph_idx in range(n_graphs):
            print(f"\n PROCESSING GRAPH {graph_idx+1}/{n_graphs}")
            
            features = batch_features[graph_idx]
            adj = batch_adj[graph_idx]
            mask = batch_masks[graph_idx]
            
            print(f"   Graph {graph_idx+1} - Valid nodes: {mask.sum()}/{len(mask)}")
            
            h = features
            all_attentions = []
            
            for i, layer in enumerate(self.layers):
                print(f"\n  GAT LAYER {i+1}/{len(self.layers)} - Graph {graph_idx+1}")
                print(f"   Input shape: {h.shape}")
                
                if return_attention:
                    h, layer_attentions = layer.forward(h, adj, return_attentions=True, training=training)
                    all_attentions.append(layer_attentions)
                else:
                    h = layer.forward(h, adj, return_attentions=False, training=training)
                
                if i < len(self.layers) - 1:
                    h = self._elu_activation(h)
            
            # Pool with mask to ignore padded nodes
            graph_embedding = self._graph_pooling(h, mask)
            batch_embeddings.append(graph_embedding)
            
            if return_attention:
                batch_all_attentions.append(all_attentions)
            
            print(f"   Graph {graph_idx+1} embedding: {graph_embedding}")
            print(f"   L2 norm: {np.linalg.norm(graph_embedding):.4f}")
        
        batch_embeddings = np.array(batch_embeddings)
        
        print(f"\n  FINAL BATCH EMBEDDINGS:")
        print(f"   Shape: {batch_embeddings.shape}")
        print(f"   Embeddings:\n{batch_embeddings}")
        print(f"   L2 norms: {[np.linalg.norm(emb) for emb in batch_embeddings]}")
        
        if return_attention:
            return batch_embeddings, batch_all_attentions
        return batch_embeddings

    def forward_batch_sequential(self, graphs_data, training=True, return_attention=True):
        Config.print_separator(" FORWARD PASS - SEQUENTIAL BATCH PROCESSING")
        
        n_graphs = len(graphs_data)
        batch_embeddings = []
        batch_all_attentions = []
        
        print(f"  SEQUENTIAL BATCH INPUT:")
        print(f"     Batch size: {n_graphs}")
        print(f"     Graph sizes: {[g['num_nodes'] for g in graphs_data]}")
        
        for graph_idx, graph_data in enumerate(graphs_data):
            print(f"\n{'🔥'} PROCESSING GRAPH {graph_idx+1}/{n_graphs}")
            
            features = graph_data['features']
            adj = graph_data['adj']
            num_nodes = graph_data['num_nodes']
            
            print(f"   Graph {graph_idx+1} - Nodes: {num_nodes}")
            print(f"   Features shape: {features.shape}")
            print(f"   Adjacency shape: {adj.shape}")
            
            h = features
            all_attentions = []
            
            for i, layer in enumerate(self.layers):
                print(f"\n  GAT LAYER {i+1}/{len(self.layers)} - Graph {graph_idx+1}")
                print(f"   Input shape: {h.shape}")
                
                if return_attention:
                    h, layer_attentions = layer.forward(h, adj, return_attentions=True, training=training)
                    all_attentions.append(layer_attentions)
                else:
                    h = layer.forward(h, adj, return_attentions=False, training=training)
                
                if i < len(self.layers) - 1:
                    h = self._elu_activation(h)
            
            graph_embedding = self._graph_pooling(h)
            batch_embeddings.append(graph_embedding)
            
            if return_attention:
                batch_all_attentions.append(all_attentions)
            
            print(f"   Graph {graph_idx+1} embedding: {graph_embedding}")
            print(f"   L2 norm: {np.linalg.norm(graph_embedding):.4f}")
        
        batch_embeddings = np.array(batch_embeddings)
        
        print(f"\n  FINAL BATCH EMBEDDINGS:")
        print(f"   Shape: {batch_embeddings.shape}")
        print(f"   Embeddings:\n{batch_embeddings}")
        print(f"   L2 norms: {[np.linalg.norm(emb) for emb in batch_embeddings]}")
        
        if return_attention:
            return batch_embeddings, batch_all_attentions
        return batch_embeddings