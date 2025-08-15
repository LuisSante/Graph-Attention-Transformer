import numpy as np
from utils.utils import init_weights, leaky_relu, softmax_with_mask
from config import Config

class GATLayer:
    def __init__(self, in_features, out_features, alpha=0.2, seed=None, dropout=0.0):
        Config.print_subsection(f" INITIALIZING GAT LAYER ") 
        print(f"  Entry: {in_features} features") 
        print(f"  Output: {out_features} features") 
        print(f"  Alpha (LeakyReLU): {alpha}") 
        print(f"  Dropout: {dropout}")

        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.dropout = dropout

        print(f"\n  Initializing transformation matrix W...") 
        self.W = init_weights(in_features, out_features, seed) 

        print(f"\n  Initializing attention 'a' vector to...") 
        self.a = init_weights(2 * out_features, 1, seed=seed).flatten()

        print(f"\n  GAT layer initialized successfully")

    def forward(self, h, adj, return_attention=False, training=True):
        Config.print_subsection(f"  FORWARD PASS - GAT LAYER")
        N = h.shape[0]

        # Step 1: Linear Transformation
        print(f"\n  STEP 1: Linear transformation")
        Wh = h @ self.W
        #Wh = h @ self.W.T  # (N, out_features)

        print(f"\n  STEP 2: Preparing concatenations")
        Wh_i = np.repeat(Wh[:, None, :], N, axis=1)  # (N, N, out_features)
        Wh_j = np.repeat(Wh[None, :, :], N, axis=0)  # (N, N, out_features)
        Wh_cat = np.concatenate([Wh_i, Wh_j], axis=-1)  # (N, N, 2*out_features)

        print(f"\n  STEP 3: Calculating attention coefficients")
        #e = Wh_cat @ self.a  # (N, N)
        e = np.sum(Wh_cat * self.a, axis=-1)  # (N, N)
        e = leaky_relu(e, self.alpha)

        print(f"\n  STEP 4: Applying mask and softmax")
        mask = adj != 0
        alpha = softmax_with_mask(e, mask, axis=1)  # (N, N)

        print(f"\n  STEP 5: Dropout (if in training)")
        if training and self.dropout > 0:
            print(f"\n  STEP 5: Applying dropout ({self.dropout})")
            dropout_mask = np.random.rand(*alpha.shape) > self.dropout
            alpha = alpha * dropout_mask
            # Renormalize after dropout
            alpha = alpha / (np.sum(alpha, axis=1, keepdims=True) + 1e-12)

        print(f"\n  STEP 6: Final aggregation")
        h_prime = alpha @ Wh  # (N, out_features)

        print(f" α @ Wh = {alpha.shape} @ {Wh.shape} = {h_prime.shape}") 
        print(f" Output h':\n{h_prime}") 
        print(f"\n  Forward pass completed")

        if return_attention:
            print("esto es alpha: ", alpha)
            return h_prime, alpha
        return h_prime

class MultiHeadGATLayer:
    def __init__(self, in_features, out_features_per_head, n_heads, concat=True, seed=None):
        Config.print_subsection(f" INITIALIZING MULTI-HEAD GAT")
        print(f"  Input: {in_features} features")
        print(f"  Number of heads: {n_heads}")
        print(f"  Features per head: {out_features_per_head}")
        print(f"  Concatenate outputs: {'Yes' if concat else 'No (average)'}")

        self.n_heads = n_heads
        self.concat = concat
        self.out_features_per_head = out_features_per_head
        
        self.heads = []
        for i in range(n_heads):
            print(f"\n  Creating attention head {i+1}/{n_heads}...")
            head_seed = None if seed is None else seed + i
            head = GATLayer(
                in_features=in_features,
                out_features=out_features_per_head,
                seed=head_seed
            )
            self.heads.append(head)

        if concat:
            self.output_dim = n_heads * out_features_per_head
            print(f"\n  Output dimension (concatenated): {self.output_dim}")
        else:
            self.output_dim = out_features_per_head
            print(f"\n  Output dimension (averaged): {self.output_dim}")


    def forward(self, h, adj, return_attentions=False, training=True):
        Config.print_separator(f" FORWARD PASS - MULTI-HEAD GAT ({self.n_heads} heads)")
        
        outputs = []
        attentions = []
        
        for i, head in enumerate(self.heads):
            print(f"\n{'='*20} PROCESSING HEAD {i+1}/{self.n_heads} {'='*20}")
            
            if return_attentions:
                out, alpha = head.forward(h, adj, return_attention=True, training=training)
                outputs.append(out)
                attentions.append(alpha)
                print(f" Head {i+1} - Output shape: {out.shape}") 
                print(f" Head {i+1} - Attention shape: {alpha.shape}")
            else:
                out = head.forward(h, adj, return_attention=False, training=training)
                outputs.append(out)
                print(f" Head {i+1} - Output shape: {out.shape}")
        
        Config.print_subsection(" ADDING HEADER OUTPUTS")
        
        if self.concat: 
            print(" Concatenating outputs...") 
            h_out = np.concatenate(outputs, axis=1) 
            print(f" Final shape: {h_out.shape}") 
        else: 
            print("Averaging outputs...") 
            h_out = np.mean(np.stack(outputs, axis=0), axis=0) 
            print(f" Final shape: {h_out.shape}") 

        print(f"\n FINAL MULTI-HEAD OUTPUT:")
        print(f"{h_out}")
        
        if return_attentions:
            return h_out, attentions
        return h_out
    
    def get_attention_weights(self, h, adj):
        _, attentions = self.forward(h, adj, return_attentions=True, training=False)
        return attentions
