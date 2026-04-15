"""Neighbor attention and Graph Transformer layer.

Implements the structure-aware attention mechanism that operates over
k-nearest neighbor graphs, following the architecture from Ingraham et al.
(2019) as used in DeepProSite.

Key operations:
  - gather_nodes: Fetch node features at neighbor indices
  - cat_neighbors_nodes: Concatenate neighbor + edge features for message passing
  - NeighborAttention: Multi-head attention over KNN neighborhoods
  - TransformerLayer: Full transformer block (attention + FFN + residual + norm)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gather_nodes(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """Gather node features at neighbor indices.

    Args:
        nodes: Node feature tensor of shape (B, N, C).
        neighbor_idx: Neighbor index tensor of shape (B, N, K).

    Returns:
        Gathered features of shape (B, N, K, C).
    """
    B, N, K = neighbor_idx.shape
    C = nodes.size(2)

    # Flatten (B, N, K) -> (B, N*K), expand to (B, N*K, C), gather, reshape.
    flat_idx = neighbor_idx.reshape(B, -1)  # (B, N*K)
    flat_idx = flat_idx.unsqueeze(-1).expand(-1, -1, C)  # (B, N*K, C)
    gathered = torch.gather(nodes, 1, flat_idx)  # (B, N*K, C)
    return gathered.reshape(B, N, K, C)


def cat_neighbors_nodes(
    h_nodes: torch.Tensor,
    h_edges: torch.Tensor,
    neighbor_idx: torch.Tensor,
) -> torch.Tensor:
    """Concatenate neighboring node features with edge features.

    For each node i and its neighbor j, produces [h_edge_ij || h_node_j].

    Args:
        h_nodes: Node hidden states, shape (B, N, D).
        h_edges: Edge hidden states, shape (B, N, K, D).
        neighbor_idx: Neighbor indices, shape (B, N, K).

    Returns:
        Concatenated features of shape (B, N, K, 2*D).
    """
    h_neighbors = gather_nodes(h_nodes, neighbor_idx)  # (B, N, K, D)
    return torch.cat([h_edges, h_neighbors], dim=-1)  # (B, N, K, 2*D)


class NeighborAttention(nn.Module):
    """Multi-head attention over k-nearest neighbors.

    Computes attention between a target node and its graph neighbors,
    incorporating edge features into keys and values.

    Args:
        hidden_dim: Hidden dimensionality (must be divisible by num_heads).
        input_dim: Input dimensionality for keys/values (edge + neighbor features).
        num_heads: Number of attention heads.
    """

    def __init__(self, hidden_dim: int, input_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_O = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        h_V: torch.Tensor,
        h_E: torch.Tensor,
        mask_attend: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute neighbor attention.

        Args:
            h_V: Node features, shape (B, N, D).
            h_E: Neighbor features (edges concatenated with neighbor nodes),
                 shape (B, N, K, D_in).
            mask_attend: Attention mask, shape (B, N, K). 1 = attend, 0 = ignore.

        Returns:
            Updated node features, shape (B, N, D).
        """
        B, N, K = h_E.shape[:3]
        H = self.num_heads
        d = self.head_dim

        # Project queries from nodes, keys/values from edge+neighbor features.
        Q = self.W_Q(h_V).reshape(B, N, 1, H, 1, d)
        K = self.W_K(h_E).reshape(B, N, K, H, d, 1)
        V = self.W_V(h_E).reshape(B, N, K, H, d)

        # Scaled dot-product attention: (B, N, K, H).
        attn_logits = torch.matmul(Q, K).reshape(B, N, K, H)
        attn_logits = attn_logits.transpose(-2, -1)  # (B, N, H, K)
        attn_logits = attn_logits / (d ** 0.5)

        if mask_attend is not None:
            mask = mask_attend.unsqueeze(2).expand(-1, -1, H, -1)  # (B, N, H, K)
            attn_logits = attn_logits.masked_fill(mask == 0, torch.finfo(attn_logits.dtype).min)

        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, N, H, K)

        if mask_attend is not None:
            attn_weights = attn_weights * mask

        # Weighted aggregation: (B, N, H, 1, K) @ (B, N, H, K, d) -> (B, N, H, 1, d)
        V_transposed = V.transpose(2, 3)  # (B, N, H, K, d)
        h_update = torch.matmul(
            attn_weights.unsqueeze(-2), V_transposed
        )  # (B, N, H, 1, d)
        h_update = h_update.reshape(B, N, self.hidden_dim)

        return self.W_O(h_update)


class TransformerLayer(nn.Module):
    """Graph Transformer layer with neighbor attention and feed-forward network.

    Architecture: LayerNorm(x + Attn(x)) → LayerNorm(x + FFN(x))

    Args:
        hidden_dim: Hidden dimensionality.
        input_dim: Input dimensionality for the attention module.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        input_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.attention = NeighborAttention(hidden_dim, input_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h_V: torch.Tensor,
        h_EV: torch.Tensor,
        mask_V: torch.Tensor | None = None,
        mask_attend: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the transformer layer.

        Args:
            h_V: Node hidden states, shape (B, N, D).
            h_EV: Edge-node concatenated features, shape (B, N, K, 2*D).
            mask_V: Node mask, shape (B, N). Applied to zero out padding nodes.
            mask_attend: Attention mask, shape (B, N, K).

        Returns:
            Updated node hidden states, shape (B, N, D).
        """
        # Self-attention with residual connection.
        dh = self.attention(h_V, h_EV, mask_attend)
        h_V = self.norm1(h_V + self.dropout(dh))

        # Feed-forward with residual connection.
        dh = self.ffn(h_V)
        h_V = self.norm2(h_V + self.dropout(dh))

        # Zero out padding positions.
        if mask_V is not None:
            h_V = h_V * mask_V.unsqueeze(-1)

        return h_V
