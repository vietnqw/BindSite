import torch
import torch.nn as nn
from .layers import EdgeFeatures, GraphTransformerLayer, gather_nodes, cat_neighbors_nodes
from ..core.config import DEFAULT_HIDDEN_DIM, DEFAULT_NUM_LAYERS, DEFAULT_NUM_HEADS, DEFAULT_DROPOUT


class DeepProSite(nn.Module):
    """Paper-faithful DeepProSite model."""

    def __init__(
        self,
        node_features=1038,
        edge_features=16,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        num_encoder_layers=DEFAULT_NUM_LAYERS,
        num_heads=DEFAULT_NUM_HEADS,
        dropout=DEFAULT_DROPOUT,
        k_neighbors=30,
        augment_eps=0.1,
    ):
        super().__init__()
        self.augment_eps = augment_eps
        self.EdgeFeatures = EdgeFeatures(edge_features, top_k=k_neighbors, augment_eps=augment_eps)
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, hidden_dim * 2, num_heads=num_heads, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, 1, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, V, mask):
        E, E_idx = self.EdgeFeatures(X, mask)

        if self.training and self.augment_eps > 0:
            V = V + 0.1 * self.augment_eps * torch.randn_like(V)

        h_V = self.W_v(V)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for layer in self.layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = layer(h_V, h_EV, mask_v=mask, mask_attend=mask_attend)

        return self.W_out(h_V).squeeze(-1)
