"""
EDGC (Equivariant Deep Graph Clustering) Partition Model for CVRP.

One-shot clustering approach combining:
1. EGNN backbone (E(2)-equivariant embeddings)
2. Dual projection heads (from SCGC/RGC for contrastive learning)
3. Learnable cluster centers (from SDCN)
4. Student-t soft cluster assignment (from SDCN)

References:
- RGC: InfoNCE contrastive loss + RL for cluster selection
- SDCN: Learnable cluster centers + Student-t distribution
- SCGC: Dual projection with noise augmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EGNNLayer(nn.Module):
    """E(2)-Equivariant GNN Layer."""

    def __init__(self, hidden_dim, act_fn='silu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.act_fn = getattr(F, act_fn)

        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, coords, e, edge_index):
        row, col = edge_index[0], edge_index[1]
        n_nodes = h.shape[0]

        coord_diff = coords[col] - coords[row]
        dist = torch.norm(coord_diff, dim=-1, keepdim=True)

        msg_input = torch.cat([h[row], h[col], dist, e], dim=-1)
        msg = self.msg_mlp(msg_input)

        coord_weights = torch.tanh(self.coord_mlp(msg))
        direction = coord_diff / (dist + 1e-8)
        coord_update = coord_weights * direction

        coord_agg = torch.zeros_like(coords)
        coord_agg.index_add_(0, row, coord_update)
        coords_new = coords + 0.1 * coord_agg

        msg_agg = torch.zeros(n_nodes, self.hidden_dim, device=h.device)
        msg_agg.index_add_(0, row, msg)

        h_new = self.node_norm(h + self.node_mlp(torch.cat([h, msg_agg], dim=-1)))
        e_new = self.edge_norm(e + self.edge_mlp(torch.cat([e, msg], dim=-1)))

        return h_new, coords_new, e_new


class EGNN(nn.Module):
    """E(2)-Equivariant GNN backbone."""

    def __init__(self, depth=12, node_feats=2, edge_feats=2, units=48):
        super().__init__()
        self.depth = depth
        self.units = units
        self.node_embed = nn.Linear(node_feats, units)
        self.edge_embed = nn.Linear(edge_feats, units)
        self.layers = nn.ModuleList([EGNNLayer(units) for _ in range(depth)])

    def forward(self, x, edge_index, edge_attr, coords):
        h = self.node_embed(x)
        e = self.edge_embed(edge_attr)
        pos = coords.clone()
        for layer in self.layers:
            h, pos, e = layer(h, pos, e, edge_index)
        return h, e


class DualProjectionHead(nn.Module):
    """Dual projection heads for contrastive learning (from SCGC/RGC)."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj1 = nn.Linear(input_dim, output_dim)
        self.proj2 = nn.Linear(input_dim, output_dim)

    def forward(self, x, is_train=True, sigma=0.01):
        z1 = F.normalize(self.proj1(x), dim=-1, p=2)
        z2 = self.proj2(x)
        if is_train:
            z2 = F.normalize(z2, dim=-1, p=2) + torch.normal(0, torch.ones_like(z2) * sigma)
        else:
            z2 = F.normalize(z2, dim=-1, p=2)
        return z1, z2


class ClusterHead(nn.Module):
    """Learnable cluster centers with Student-t assignment (from SDCN)."""

    def __init__(self, embedding_dim, max_clusters=50, v=1.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_clusters = max_clusters
        self.v = v
        self.cluster_centers = nn.Parameter(torch.Tensor(max_clusters, embedding_dim))
        nn.init.xavier_normal_(self.cluster_centers)

    def forward(self, z, n_clusters=None):
        if n_clusters is None:
            n_clusters = self.max_clusters
        centers = self.cluster_centers[:n_clusters]

        if z.dim() == 2:
            distances = torch.sum((z.unsqueeze(1) - centers.unsqueeze(0)) ** 2, dim=-1)
        else:
            distances = torch.sum((z.unsqueeze(2) - centers.unsqueeze(0).unsqueeze(0)) ** 2, dim=-1)

        q = 1.0 / (1.0 + distances / self.v)
        q = q ** ((self.v + 1.0) / 2.0)
        q = q / (q.sum(dim=-1, keepdim=True) + 1e-8)
        return q, distances

    def get_target_distribution(self, q):
        """Compute auxiliary target distribution P for self-training."""
        weight = q ** 2 / (q.sum(dim=-2, keepdim=True) + 1e-8)
        p = weight / (weight.sum(dim=-1, keepdim=True) + 1e-8)
        return p


class ClusterPartitionModel(nn.Module):
    """
    EDGC Partition Model for CVRP.

    Combines EGNN backbone with clustering heads for one-shot partitioning.
    """

    def __init__(self, units=48, node_feats=2, edge_feats=2, depth=12,
                 projection_dim=128, max_clusters=50):
        super().__init__()
        self.egnn = EGNN(depth=depth, node_feats=node_feats, edge_feats=edge_feats, units=units)
        self.dual_proj = DualProjectionHead(units, projection_dim)
        self.cluster_head = ClusterHead(projection_dim, max_clusters)
        self.k_predictor = nn.Sequential(
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, max_clusters)
        )
        self.node_emb = None
        self.edge_emb = None

    def forward(self, pyg, n_clusters=None, is_train=True, sigma=0.01):
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr
        coords = pyg.pos if hasattr(pyg, 'pos') else pyg.x[:, :2]

        self.node_emb, self.edge_emb = self.egnn(x, edge_index, edge_attr, coords)
        z1, z2 = self.dual_proj(self.node_emb, is_train=is_train, sigma=sigma)

        global_emb = self.node_emb.mean(dim=0)
        k_logits = self.k_predictor(global_emb)

        if n_clusters is None:
            n_clusters = max(torch.argmax(k_logits).item() + 1, 2)

        q, distances = self.cluster_head(z1, n_clusters)
        return q, z1, z2, k_logits, distances

    def get_hard_assignment(self, q, sample=False):
        if sample:
            gumbel = -torch.log(-torch.log(torch.rand_like(q) + 1e-8) + 1e-8)
            logits = torch.log(q + 1e-8) + gumbel
            labels = F.softmax(logits, dim=-1).argmax(dim=-1)
        else:
            labels = q.argmax(dim=-1)
        log_prob = torch.log(q.gather(1, labels.unsqueeze(1)) + 1e-8).squeeze()
        return labels, log_prob
