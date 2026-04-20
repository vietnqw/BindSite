import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gather_nodes(nodes, neighbor_idx):
    # Features [B, N, C] at Neighbor indices [B, N, K] -> [B, N, K, C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, edge_idx):
    h_nodes = gather_nodes(h_nodes, edge_idx)
    return torch.cat([h_neighbors, h_nodes], -1)


class Normalize(nn.Module):
    """LayerNorm-style normalization over the last feature dimension."""

    def __init__(self, features, epsilon=1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size(0)
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        self.num_embeddings = num_embeddings

    def forward(self, edge_idx):
        n_nodes = edge_idx.size(1)
        ii = torch.arange(n_nodes, dtype=torch.float32, device=edge_idx.device).view((1, -1, 1))
        d = (edge_idx.float() - ii).unsqueeze(-1)
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32, device=edge_idx.device)
            * -(np.log(10000.0) / self.num_embeddings)
        )
        angles = d * frequency.view((1, 1, 1, -1))
        return torch.cat((torch.cos(angles), torch.sin(angles)), -1)


class EdgeFeatures(nn.Module):
    """Paper-faithful geometric edge features and projection."""

    def __init__(
        self,
        edge_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=30,
        augment_eps=0.0,
    ):
        super().__init__()
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.PE = PositionalEncodings(num_positional_embeddings)
        self.edge_embedding = nn.Linear(num_positional_embeddings + num_rbf + 7, edge_features, bias=True)
        self.norm_edges = Normalize(edge_features)

    def _dist(self, x, mask, eps=1e-6):
        mask_2d = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(x, 1) - torch.unsqueeze(x, 2)
        D = mask_2d * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2d) * D_max
        D_neighbors, edge_idx = torch.topk(D_adjust, self.top_k, dim=-1, largest=False)
        return D_neighbors, edge_idx

    def _rbf(self, D):
        D_min, D_max, D_count = 0.0, 20.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device).view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        return torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)

    def _quaternions(self, R):
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(
            torch.abs(
                1
                + torch.stack(
                    [
                        Rxx - Ryy - Rzz,
                        -Rxx + Ryy - Rzz,
                        -Rxx - Ryy + Rzz,
                    ],
                    -1,
                )
            )
        )
        get_r = lambda i, j: R[:, :, :, i, j]
        signs = torch.sign(
            torch.stack(
                [
                    get_r(2, 1) - get_r(1, 2),
                    get_r(0, 2) - get_r(2, 0),
                    get_r(1, 0) - get_r(0, 1),
                ],
                -1,
            )
        )
        xyz = signs * magnitudes
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.0
        Q = torch.cat((xyz, w), -1)
        return F.normalize(Q, dim=-1)

    def _orientations(self, x, edge_idx):
        dX = x[:, 1:, :] - x[:, :-1, :]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2, dim=-1)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0, 0, 1, 2), "constant", 0)

        O_neighbors = gather_nodes(O, edge_idx)
        X_neighbors = gather_nodes(x, edge_idx)
        O = O.view(list(O.shape[:2]) + [3, 3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3, 3])

        dX = X_neighbors - x.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1, -2), O_neighbors)
        Q = self._quaternions(R)
        return torch.cat((dU, Q), dim=-1)

    def forward(self, x, mask):
        if self.training and self.augment_eps > 0:
            x = x + self.augment_eps * torch.randn_like(x)

        D_neighbors, edge_idx = self._dist(x, mask)
        rbf = self._rbf(D_neighbors)
        o_features = self._orientations(x, edge_idx)
        e_positional = self.PE(edge_idx)

        E = torch.cat((e_positional, rbf, o_features), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, edge_idx


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)

    def forward(self, h_v):
        return self.W_out(F.relu(self.W_in(h_v)))


class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_in, num_hidden, bias=False)
        self.W_V = nn.Linear(num_in, num_hidden, bias=False)
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def _masked_softmax(self, attend_logits, mask_attend, dim=-1):
        negative_inf = torch.finfo(attend_logits.dtype).min
        attend_logits = torch.where(mask_attend > 0, attend_logits, negative_inf)
        attend = F.softmax(attend_logits, dim)
        return mask_attend * attend

    def forward(self, h_v, h_e, mask_attend=None):
        n_batch, n_nodes, n_neighbors = h_e.shape[:3]
        n_heads = self.num_heads
        d = int(self.num_hidden / n_heads)

        Q = self.W_Q(h_v).view([n_batch, n_nodes, 1, n_heads, 1, d])
        K = self.W_K(h_e).view([n_batch, n_nodes, n_neighbors, n_heads, d, 1])
        V = self.W_V(h_e).view([n_batch, n_nodes, n_neighbors, n_heads, d])

        attend_logits = torch.matmul(Q, K).view([n_batch, n_nodes, n_neighbors, n_heads]).transpose(-2, -1)
        attend_logits = attend_logits / np.sqrt(d)

        if mask_attend is not None:
            mask = mask_attend.unsqueeze(2).expand(-1, -1, n_heads, -1)
            attend = self._masked_softmax(attend_logits, mask)
        else:
            attend = F.softmax(attend_logits, -1)

        h_v_update = torch.matmul(attend.unsqueeze(-2), V.transpose(2, 3))
        h_v_update = h_v_update.view([n_batch, n_nodes, self.num_hidden])
        return self.W_O(h_v_update)


class GraphTransformerLayer(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([Normalize(num_hidden) for _ in range(2)])
        self.attention = NeighborAttention(num_hidden, num_in, num_heads=num_heads)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_v, h_e, mask_v=None, mask_attend=None):
        dh = self.attention(h_v, h_e, mask_attend)
        h_v = self.norm[0](h_v + self.dropout(dh))

        dh = self.dense(h_v)
        h_v = self.norm[1](h_v + self.dropout(dh))

        if mask_v is not None:
            h_v = mask_v.unsqueeze(-1) * h_v
        return h_v
