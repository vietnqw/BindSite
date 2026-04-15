"""Geometric edge feature computation for protein graphs.

Computes edge features from Cα atom coordinates following
Ingraham et al. (2019), including:
  - Radial basis functions (RBF) on pairwise distances
  - Relative positional encodings (sinusoidal)
  - Relative direction and orientation (quaternion) features
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bindsite.model.attention import gather_nodes


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding based on sequence separation.

    Encodes the sequence gap (i - j) between residues using sinusoidal
    functions at multiple frequencies.

    Args:
        num_embeddings: Dimensionality of the positional encoding.
    """

    def __init__(self, num_embeddings: int = 16) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings

    def forward(self, E_idx: torch.Tensor) -> torch.Tensor:
        """Compute positional encodings for neighbor indices.

        Args:
            E_idx: Neighbor index tensor, shape (B, N, K).

        Returns:
            Positional encodings of shape (B, N, K, num_embeddings).
        """
        device = E_idx.device
        N = E_idx.size(1)

        # Sequence positions: (1, N, 1).
        ii = torch.arange(N, dtype=torch.float32, device=device).reshape(1, -1, 1)

        # Sequence gap: (B, N, K, 1).
        d = (E_idx.float() - ii).unsqueeze(-1)

        # Sinusoidal frequencies.
        freq = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32, device=device)
            * -(np.log(10000.0) / self.num_embeddings)
        )
        freq = freq.reshape(1, 1, 1, -1)

        angles = d * freq
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


class EdgeFeatures(nn.Module):
    """Geometric edge feature computation for protein structure graphs.

    Constructs a k-nearest neighbor graph from Cα coordinates and computes
    rich edge features encoding distance, direction, and orientation.

    Edge features consist of:
      - Positional encodings (sequence gap, 16-d)
      - Radial basis functions (distance, 16-d)
      - Orientation features (direction + quaternion, 7-d)

    Total raw features: 16 + 16 + 7 = 39, projected to edge_features dims.

    Args:
        edge_features: Output edge feature dimensionality.
        num_positional: Dimensionality of positional encodings.
        num_rbf: Number of radial basis functions.
        k_neighbors: Number of nearest neighbors.
        augment_eps: Scale of coordinate noise for data augmentation.
    """

    def __init__(
        self,
        edge_features: int = 16,
        num_positional: int = 16,
        num_rbf: int = 16,
        k_neighbors: int = 30,
        augment_eps: float = 0.0,
    ) -> None:
        super().__init__()
        self.k_neighbors = k_neighbors
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf

        self.positional_encoding = PositionalEncoding(num_positional)
        self.edge_projection = nn.Linear(
            num_positional + num_rbf + 7, edge_features, bias=True
        )
        self.edge_norm = nn.LayerNorm(edge_features)

    def _compute_knn(
        self, X: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find k-nearest neighbors based on Euclidean distance.

        Args:
            X: Cα coordinates, shape (B, N, 3).
            mask: Sequence mask, shape (B, N).

        Returns:
            Tuple of:
              - D_neighbors: Distances to k-nearest neighbors, shape (B, N, K).
              - E_idx: Indices of k-nearest neighbors, shape (B, N, K).
        """
        # Pairwise distance matrix.
        mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, N, N)
        dX = X.unsqueeze(1) - X.unsqueeze(2)  # (B, N, N, 3)
        D = mask_2D * torch.sqrt((dX ** 2).sum(dim=3) + 1e-6)  # (B, N, N)

        # Set non-masked distances to max so they're not selected as neighbors.
        D_max = D.max(dim=-1, keepdim=True).values
        D_adjusted = D + (1.0 - mask_2D) * D_max

        # Select k smallest distances (including self).
        D_neighbors, E_idx = torch.topk(
            D_adjusted, self.k_neighbors, dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D: torch.Tensor) -> torch.Tensor:
        """Compute radial basis function features.

        Uses evenly-spaced Gaussian RBFs from 0 to 20 Å.

        Args:
            D: Distance tensor, shape (B, N, K).

        Returns:
            RBF features of shape (B, N, K, num_rbf).
        """
        D_min, D_max = 0.0, 20.0
        mu = torch.linspace(D_min, D_max, self.num_rbf, device=D.device)
        mu = mu.reshape(1, 1, 1, -1)
        sigma = (D_max - D_min) / self.num_rbf

        return torch.exp(-((D.unsqueeze(-1) - mu) / sigma) ** 2)

    def _compute_orientations(
        self, X: torch.Tensor, E_idx: torch.Tensor
    ) -> torch.Tensor:
        """Compute relative orientation features between residue pairs.

        Extracts local coordinate frames from backbone geometry, then computes:
          - Relative direction (3-d): normalized displacement in local frame
          - Quaternion (4-d): rotation between local frames

        Args:
            X: Cα coordinates, shape (B, N, 3).
            E_idx: Neighbor indices, shape (B, N, K).

        Returns:
            Orientation features of shape (B, N, K, 7).
        """
        # Unit vectors along backbone.
        dX = X[:, 1:] - X[:, :-1]
        U = F.normalize(dX, dim=-1)

        # Backbone normals from cross products.
        u_prev = U[:, :-2]  # (B, N-3, 3)
        u_next = U[:, 1:-1]  # (B, N-3, 3)
        n = F.normalize(torch.cross(u_prev, u_next, dim=-1), dim=-1)

        # Bisector and local frame.
        b = F.normalize(u_prev - u_next, dim=-1)
        O = torch.stack([b, n, torch.cross(b, n, dim=-1)], dim=2)  # (B, N-3, 3, 3)
        O = O.reshape(list(O.shape[:2]) + [9])

        # Pad first and last 2 residues with zeros.
        O = F.pad(O, (0, 0, 1, 2), value=0.0)  # (B, N, 9)

        # Gather neighbor orientations and coordinates.
        O_neighbors = gather_nodes(O, E_idx)  # (B, N, K, 9)
        X_neighbors = gather_nodes(X, E_idx)  # (B, N, K, 3)

        # Reshape to rotation matrices.
        O_mat = O.reshape(list(O.shape[:2]) + [3, 3])  # (B, N, 3, 3)
        O_nb_mat = O_neighbors.reshape(list(O_neighbors.shape[:3]) + [3, 3])

        # Relative direction in local frame.
        dX_nb = X_neighbors - X.unsqueeze(2)  # (B, N, K, 3)
        dU = torch.matmul(O_mat.unsqueeze(2), dX_nb.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)  # (B, N, K, 3)

        # Relative rotation → quaternion.
        R = torch.matmul(O_mat.unsqueeze(2).transpose(-1, -2), O_nb_mat)
        Q = self._rotation_to_quaternion(R)  # (B, N, K, 4)

        return torch.cat([dU, Q], dim=-1)  # (B, N, K, 7)

    @staticmethod
    def _rotation_to_quaternion(R: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrices to quaternions.

        Args:
            R: Rotation matrices, shape (..., 3, 3).

        Returns:
            Quaternions, shape (..., 4).
        """
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)

        magnitudes = 0.5 * torch.sqrt(
            torch.abs(
                1 + torch.stack([Rxx - Ryy - Rzz, -Rxx + Ryy - Rzz, -Rxx - Ryy + Rzz], dim=-1)
            )
        )
        signs = torch.sign(
            torch.stack(
                [R[..., 2, 1] - R[..., 1, 2],
                 R[..., 0, 2] - R[..., 2, 0],
                 R[..., 1, 0] - R[..., 0, 1]],
                dim=-1,
            )
        )
        xyz = signs * magnitudes
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.0

        Q = torch.cat([xyz, w], dim=-1)
        return F.normalize(Q, dim=-1)

    def forward(
        self, X: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute edge features for the protein graph.

        Args:
            X: Cα coordinates, shape (B, N, 3).
            mask: Sequence mask, shape (B, N).

        Returns:
            Tuple of:
              - E: Edge features, shape (B, N, K, edge_features).
              - E_idx: Neighbor indices, shape (B, N, K).
        """
        # Optional coordinate augmentation during training.
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        # Build KNN graph.
        D_neighbors, E_idx = self._compute_knn(X, mask)

        # Compute feature components.
        rbf = self._rbf(D_neighbors)  # (B, N, K, num_rbf)
        pos_enc = self.positional_encoding(E_idx)  # (B, N, K, num_pos)
        orient = self._compute_orientations(X, E_idx)  # (B, N, K, 7)

        # Concatenate and project.
        E_raw = torch.cat([pos_enc, rbf, orient], dim=-1)
        E = self.edge_projection(E_raw)
        E = self.edge_norm(E)

        return E, E_idx
