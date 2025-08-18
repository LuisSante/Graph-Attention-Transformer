import numpy as np
from models.gat_model import GATModel
from models.mlp import MLP
from config import Config

class GatMLP:
    def __init__(self, gat_config, mlp_config, seed=None):
        Config.print_separator("INITIALIZING GAT-MLP MODEL", "=", 100)
        
        self.seed = seed
        
        print("\n INITIALIZING GAT COMPONENT:")
        self.gat_model = GATModel(
            input_dim=gat_config['input_dim'],
            hidden_dims=gat_config['hidden_dims'],
            n_heads=gat_config['n_heads'],
            dropout=gat_config.get('dropout', 0.0),
            alpha=gat_config.get('alpha', 0.2),
            seed=seed,
            pooling_method=gat_config.get('pooling_method', 'mean'),
            batch_processing=gat_config.get('batch_processing', 'padding')
        )
        
        print("\n INITIALIZING MLP COMPONENT:")
        self.mlp_classifier = MLP(
            input_dim=self.gat_model.output_dim,  # Connect GAT output to MLP input
            hidden_dims=mlp_config['hidden_dims'],
            num_classes=mlp_config['num_classes'],
            dropout=mlp_config.get('dropout', 0.0),
            activation=mlp_config.get('activation', 'relu'),
            seed=seed
        )
        
        print(f"\n GAT-MLP Model initialized successfully!")
        print(f"   GAT output dim: {self.gat_model.output_dim}")
        print(f"   MLP input dim: {self.mlp_classifier.input_dim}")
        print(f"   Final classes: {self.mlp_classifier.num_classes}")
    
    def forward(self, features, adj, training=True, return_attention=False, 
                return_embeddings=False):
        Config.print_separator("FULL FORWARD PASS: GAT → MLP", "=", 100)
        
        print("Phase 1: GAT Processing...")
        if return_attention:
            graph_embedding, attentions = self.gat_model.forward(
                features, adj, training=training, return_attention=True
            )
        else:
            graph_embedding = self.gat_model.forward(
                features, adj, training=training, return_attention=False
            )
            attentions = None
        
        print("\nPhase 2: MLP Classification...")
        predictions = self.mlp_classifier.forward(graph_embedding, training=training)
        
        print(f"\n FINAL RESULTS:")
        print(f"   Graph embedding shape: {graph_embedding.shape}")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Predicted class: {np.argmax(predictions)}")
        print(f"   Class probabilities: {predictions}")
        
        results = {'predictions': predictions}
        if return_embeddings:
            results['embeddings'] = graph_embedding
        if return_attention:
            results['attentions'] = attentions
        
        return results
    
    def forward_batch(self, batch_data, training=True, return_attention=False, return_embeddings=False):
        Config.print_separator("BATCH FORWARD PASS: GAT → MLP", "=", 100)
        
        if isinstance(batch_data, dict) and 'adj' in batch_data:
            # Padded batch format
            batch_features = batch_data['features']
            batch_adj = batch_data['adj']
            batch_masks = batch_data['masks']
            
            print("Phase 1: GAT Batch Processing (Padded)...")
            if return_attention:
                batch_embeddings, batch_attentions = self.gat_model.forward_batch_padded(
                    batch_features, batch_adj, batch_masks, 
                    training=training, return_attention=True
                )
            else:
                batch_embeddings = self.gat_model.forward_batch_padded(
                    batch_features, batch_adj, batch_masks, 
                    training=training, return_attention=False
                )
                batch_attentions = None
                
        else:
            print("Phase 1: GAT Batch Processing (Sequential)...")
            if return_attention:
                batch_embeddings, batch_attentions = self.gat_model.forward_batch_sequential(
                    batch_data, training=training, return_attention=True
                )
            else:
                batch_embeddings = self.gat_model.forward_batch_sequential(
                    batch_data, training=training, return_attention=False
                )
                batch_attentions = None
        
        print("\nPhase 2: MLP Batch Classification...")
        batch_predictions = self.mlp_classifier.forward(batch_embeddings, training=training)
        
        predicted_classes = np.argmax(batch_predictions, axis=1)
        
        print(f"\n BATCH RESULTS:")
        print(f"   Batch size: {len(batch_embeddings)}")
        print(f"   Graph embeddings shape: {batch_embeddings.shape}")
        print(f"   Predictions shape: {batch_predictions.shape}")
        print(f"   Predicted classes: {predicted_classes}")
        print(f"   Max probabilities: {np.max(batch_predictions, axis=1)}")
        
        results = {'predictions': batch_predictions, 'predicted_classes': predicted_classes}
        if return_embeddings:
            results['embeddings'] = batch_embeddings
        if return_attention:
            results['attentions'] = batch_attentions
        
        return results