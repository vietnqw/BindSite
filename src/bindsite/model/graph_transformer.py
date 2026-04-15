"""Graph Transformer model for protein binding site prediction.

The main model that combines geometric edge features with node features
(ProtT5 + DSSP) through a stack of structure-aware transformer layers
to produce per-residue binding site predictions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bindsite.model.attention import TransformerLayer, cat_neighbors_nodes, gather_nodes
from bindsite.model.edge_features import EdgeFeatures


class GraphTransformer(nn.Module):
    """Graph Transformer for protein binding site prediction.

    Architecture:
        1. Compute geometric edge features from Cα coordinates (KNN graph)
        2. Project node features (ProtT5 + DSSP) and edge features to hidden dim
        3. Stack of N Graph Transformer layers with neighbor attention
        4. Linear projection to per-residue logits

    Args:
        node_features: Input node feature dimensionality (default: 1038).
        edge_features: Edge feature embedding dimensionality (default: 16).
        hidden_dim: Hidden dimensionality throughout the model (default: 64).
        num_encoder_layers: Number of transformer layers (default: 4).
        num_attention_heads: Number of attention heads (default: 4).
        k_neighbors: Number of nearest neighbors in KNN graph (default: 30).
        augment_eps: Gaussian noise scale for data augmentation (default: 0.1).
        dropout: Dropout probability (default: 0.3).
    """

    def __init__(
        self,
        node_features: int = 1038,
        edge_features: int = 16,
        hidden_dim: int = 64,
        num_encoder_layers: int = 4,
        num_attention_heads: int = 4,
        k_neighbors: int = 30,
        augment_eps: float = 0.1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.augment_eps = augment_eps

        # Edge feature computation from 3D coordinates.
        self.edge_features = EdgeFeatures(
            edge_features=edge_features,
            k_neighbors=k_neighbors,
            augment_eps=augment_eps,
        )

        # Input projections.
        self.node_projection = nn.Linear(node_features, hidden_dim)
        self.edge_projection = nn.Linear(edge_features, hidden_dim)

        # Transformer encoder stack.
        # Input dim for attention is 2*hidden_dim (edge + neighbor node features).
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(
                hidden_dim=hidden_dim,
                input_dim=hidden_dim * 2,
                num_heads=num_attention_heads,
                dropout=dropout,
            )
            for _ in range(num_encoder_layers)
        ])

        # Output projection: hidden_dim -> 1 logit per residue.
        self.output_projection = nn.Linear(hidden_dim, 1)

        # Initialize weights.
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization for all weight matrices."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        coords: torch.Tensor,
        node_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass producing per-residue binding site logits.

        Args:
            coords: Cα coordinates, shape (B, L, 3).
            node_features: Pre-computed node features (ProtT5 + DSSP),
                shape (B, L, node_features).
            mask: Sequence mask, shape (B, L). 1 for real residues, 0 for padding.

        Returns:
            Per-residue logits (before sigmoid), shape (B, L).
        """
        # Step 1: Compute geometric edge features and KNN graph.
        E, E_idx = self.edge_features(coords, mask)  # E: (B,L,K,d_e), E_idx: (B,L,K)

        # Step 2: Optional node feature augmentation during training.
        if self.training and self.augment_eps > 0:
            node_features = node_features + 0.1 * self.augment_eps * torch.randn_like(
                node_features
            )

        # Step 3: Project inputs to hidden dimension.
        h_V = self.node_projection(node_features)  # (B, L, D)
        h_E = self.edge_projection(E)  # (B, L, K, D)

        # Step 4: Compute attention mask from node mask.
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend  # (B, L, K)

        # Step 5: Apply transformer layers.
        for layer in self.encoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)  # (B, L, K, 2*D)
            h_V = layer(h_V, h_EV, mask_V=mask, mask_attend=mask_attend)

        # Step 6: Project to per-residue logits.
        logits = self.output_projection(h_V).squeeze(-1)  # (B, L)

        return logits
