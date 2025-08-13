import numpy as np
from utils.utils import init_weights, leaky_relu, softmax_with_mask

class GATLayer:
    def __init__(self, in_features, out_features, alpha=0.2, seed=None, dropout=0.0):
        self.in_features = in_features
        self.out_features = out_features
        self.W = init_weights(in_features, out_features)
        self.a = init_weights(2 * out_features, 1, seed=seed).flatten()
        self.alpha = alpha
        self.dropout = dropout

    def forward(self, h, adj, return_attention=False, training=True):
        N = h.shape[0]
        Wh = h @ self.W
        #Wh = h @ self.W.T  # (N, out_features)

        Wh_i = np.repeat(Wh[:, None, :], N, axis=1)  # (N, N, out_features)
        Wh_j = np.repeat(Wh[None, :, :], N, axis=0)  # (N, N, out_features)
        Wh_cat = np.concatenate([Wh_i, Wh_j], axis=-1)  # (N, N, 2*out_features)

        #e = Wh_cat @ self.a  # (N, N)
        e = np.sum(Wh_cat * self.a, axis=-1)  # (N, N)
        e = leaky_relu(e, self.alpha)

        mask = adj != 0
        alpha = softmax_with_mask(e, mask, axis=1)  # (N, N)

        if training and self.dropout > 0:
            dropout_mask = np.random.rand(*alpha.shape) > self.dropout
            alpha = alpha * dropout_mask
            # Renormalize after dropout
            alpha = alpha / (np.sum(alpha, axis=1, keepdims=True) + 1e-12)

        h_prime = alpha @ Wh  # (N, out_features)

        if return_attention:
            return h_prime, alpha
        return h_prime

class MultiHeadGATLayer:
    def __init__(self, in_features, out_features_per_head, n_heads, concat=True, seed=None):
        self.n_heads = n_heads
        self.concat = concat
        self.out_features_per_head = out_features_per_head
        
        self.heads = []
        for i in range(n_heads):
            head_seed = None if seed is None else seed + i
            head = GATLayer(
                in_features=in_features,
                out_features=out_features_per_head,
                seed=head_seed
            )
            self.heads.append(head)

        if concat:
            self.output_dim = n_heads * out_features_per_head
        else:
            self.output_dim = out_features_per_head


    def forward(self, h, adj, return_attentions=False, training=True):
        outputs = []
        attentions = []
        for head in self.heads:
            if return_attentions:
                out, alpha = head.forward(h, adj, return_attention=True, training=training)
                outputs.append(out)
                attentions.append(alpha)
            else:
                out = head.forward(h, adj, return_attention=False, training=training)
                outputs.append(out)

        if self.concat:
            h_out = np.concatenate(outputs, axis=1)
        else:
            h_out = np.mean(np.stack(outputs, axis=0), axis=0)

        if return_attentions:
            return h_out, attentions
        return h_out
    
    def get_attention_weights(self, h, adj):
        _, attentions = self.forward(h, adj, return_attentions=True, training=False)
        return attentions
