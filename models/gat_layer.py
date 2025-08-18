import numpy as np
from utils.utils import init_weights, leaky_relu, softmax_with_mask
from config import Config

class GATAttentionHead:
    def __init__(self, in_features, out_features, alpha=0.2, seed=None):
        Config.print_subsection(f" INITIALIZING GAT LAYER ") 
        print(f"  Entry: {in_features} features") 
        print(f"  Output: {out_features} features") 
        print(f"  Alpha (LeakyReLU): {alpha}") 

        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = init_weights(in_features, out_features, seed) 
        print(f"\n  Initializing transformation matrix W...") 

        print(f"\n  Initializing attention 'a' vector to...") 
        self.a = init_weights(2 * out_features, 1, seed=seed).flatten()

        print(f"\n  GAT layer initialized successfully")

    def forward(self, h, adj, return_attention=False):
        print(f"        Processing head: {h.shape} -> ", end="")
        
        # Linear transformation
        Wh = np.dot(h, self.W)  # [N, out_features]
        N = Wh.shape[0]
        
        # Self-attention mechanism
        Wh_repeated_in_chunks = np.repeat(Wh, N, axis=0).reshape(N, N, -1)
        Wh_repeated_alternating = np.tile(Wh, (N, 1)).reshape(N, N, -1)
        
        all_combinations_matrix = np.concatenate([Wh_repeated_in_chunks, Wh_repeated_alternating], axis=2)
        
        e = np.dot(all_combinations_matrix.reshape(-1, 2 * self.out_features), self.a).reshape(N, N)
        e = leaky_relu(e, self.alpha)
        
        # Apply attention only to connected nodes
        attention = softmax_with_mask(e, adj.astype(bool))
        h_prime = np.dot(attention, Wh)
        
        print(f"{h_prime.shape}")
        
        if return_attention:
            return h_prime, attention
        return h_prime
        
class MultiHeadGATLayer:
    def __init__(self, in_features, out_features_per_head, n_heads, 
                 concat=True, alpha=0.2, seed=None):
        print(f"     Multi-head GAT Layer: {in_features} -> {n_heads}×{out_features_per_head}")
        self.in_features = in_features
        self.out_features_per_head = out_features_per_head
        self.n_heads = n_heads
        self.concat = concat
        self.alpha = alpha
        
        # Create attention heads
        self.attention_heads = []
        for i in range(n_heads):
            head_seed = seed + i if seed is not None else None
            head = GATAttentionHead(in_features, out_features_per_head, alpha, head_seed)
            self.attention_heads.append(head)
            
        self.output_dim = n_heads * out_features_per_head if concat else out_features_per_head

    def forward(self, h, adj, return_attentions=False, training=True):
        print(f"    Processing {self.n_heads} attention heads...")
        head_outputs = []
        head_attentions = []
        
        for i, head in enumerate(self.attention_heads):
            print(f"      Head {i+1}/{self.n_heads}: ", end="")
            if return_attentions:
                head_out, head_att = head.forward(h, adj, return_attention=True)
                head_attentions.append(head_att)
            else:
                head_out = head.forward(h, adj, return_attention=False)
            head_outputs.append(head_out)
        
        if self.concat:
            output = np.concatenate(head_outputs, axis=1)
            print(f"    Concatenated output: {output.shape}")
        else:
            output = np.mean(head_outputs, axis=0)
            print(f"    Averaged output: {output.shape}")
        
        if return_attentions:
            return output, head_attentions
        return output
