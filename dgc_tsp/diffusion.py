"""
Hierarchical Diffusion for TSP

Two-level diffusion process:
- Coarse level: Inter-cluster routing (which clusters connect)
- Fine level: Intra-cluster routing (routing within each cluster)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np


class CategoricalDiffusionProcess:
    """
    Categorical diffusion process for binary edge prediction.
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 1.5,
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """Linear noise schedule."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Integral of beta from 0 to t."""
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2

    def transition_probability(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute P(X_t | X_0) for binary case."""
        integral = self.integral_beta(t)
        decay = torch.exp(-2 * integral)
        p_same = 0.5 + 0.5 * decay
        p_diff = 0.5 - 0.5 * decay
        return p_same, p_diff

    def sample_forward(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Sample X_t given X_0."""
        p_same, p_diff = self.transition_probability(t)

        # Expand to match x0 shape
        while p_diff.dim() < x0.dim():
            p_diff = p_diff.unsqueeze(-1)

        # Flip with probability p_diff
        uniform = torch.rand_like(x0.float())
        flip_mask = uniform < p_diff
        xt = torch.where(flip_mask, 1 - x0, x0)

        return xt.long()


class ScoreNetwork(nn.Module):
    """
    Score network for predicting clean adjacency from noisy state.
    """

    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Edge embedding
        self.edge_embed = nn.Linear(1, hidden_dim)

        # Node feature projection
        self.node_proj = nn.Linear(node_dim, hidden_dim)

        # Transformer layers for edge prediction
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),  # Binary classification
        )

    def forward(
        self,
        h: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict clean adjacency from noisy state.

        Args:
            h: Node embeddings (batch, n, node_dim)
            xt: Noisy adjacency (batch, n, n)
            t: Time (batch,)

        Returns:
            Logits for clean adjacency (batch, n, n, 2)
        """
        batch_size, n, _ = h.shape

        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1))  # (batch, hidden)

        # Node features
        h_proj = self.node_proj(h)  # (batch, n, hidden)

        # Add time embedding to nodes
        h_with_time = h_proj + t_emb.unsqueeze(1)

        # Apply transformer layers
        for layer in self.layers:
            h_with_time = layer(h_with_time)

        # Edge prediction: combine node pairs
        h_i = h_with_time.unsqueeze(2).expand(-1, -1, n, -1)  # (batch, n, n, hidden)
        h_j = h_with_time.unsqueeze(1).expand(-1, n, -1, -1)  # (batch, n, n, hidden)

        # Current edge state
        edge_emb = self.edge_embed(xt.unsqueeze(-1).float())  # (batch, n, n, hidden)

        # Combine and predict
        edge_input = torch.cat([h_i, h_j, edge_emb], dim=-1)
        logits = self.output_head(edge_input)

        return logits


class HierarchicalDiffusion(nn.Module):
    """
    Hierarchical diffusion for TSP with coarse and fine levels.

    Coarse level: Predicts inter-cluster connections
    Fine level: Predicts intra-cluster routing
    """

    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        num_inference_steps: int = 50,
    ):
        super().__init__()
        self.num_inference_steps = num_inference_steps

        # Coarse-level score network (cluster connections)
        self.coarse_score_net = ScoreNetwork(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        # Fine-level score network (intra-cluster routing)
        self.fine_score_net = ScoreNetwork(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        # Cross-level message passing
        self.coarse_to_fine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        # Diffusion process
        self.diffusion = CategoricalDiffusionProcess()

    def get_cluster_adjacency(
        self,
        node_adj: torch.Tensor,
        assignments: torch.Tensor,
        num_clusters: int,
    ) -> torch.Tensor:
        """
        Convert node-level adjacency to cluster-level adjacency.

        Args:
            node_adj: Node adjacency (batch, n, n)
            assignments: Cluster assignments (batch, n)
            num_clusters: Number of clusters

        Returns:
            Cluster adjacency (batch, k, k)
        """
        batch_size, n, _ = node_adj.shape
        device = node_adj.device

        cluster_adj = torch.zeros(batch_size, num_clusters, num_clusters, device=device)

        for b in range(batch_size):
            for i in range(n):
                for j in range(n):
                    if node_adj[b, i, j] > 0.5:
                        ci = assignments[b, i]
                        cj = assignments[b, j]
                        if ci != cj:  # Inter-cluster edge
                            cluster_adj[b, ci, cj] = 1
                            cluster_adj[b, cj, ci] = 1

        return cluster_adj

    def compute_cluster_embeddings(
        self,
        h: torch.Tensor,
        assignments: torch.Tensor,
        num_clusters: int,
    ) -> torch.Tensor:
        """
        Compute cluster-level embeddings by aggregating node embeddings.

        Args:
            h: Node embeddings (batch, n, d)
            assignments: Cluster assignments (batch, n)
            num_clusters: Number of clusters

        Returns:
            Cluster embeddings (batch, k, d)
        """
        batch_size, n, d = h.shape
        device = h.device

        cluster_h = torch.zeros(batch_size, num_clusters, d, device=device)

        for b in range(batch_size):
            for k in range(num_clusters):
                mask = (assignments[b] == k)
                if mask.sum() > 0:
                    cluster_h[b, k] = h[b, mask].mean(dim=0)

        return cluster_h

    def coarse_diffusion_loss(
        self,
        cluster_h: torch.Tensor,
        cluster_adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute coarse-level diffusion loss.

        Args:
            cluster_h: Cluster embeddings (batch, k, d)
            cluster_adj: Ground truth cluster adjacency (batch, k, k)

        Returns:
            Loss scalar
        """
        batch_size = cluster_h.shape[0]
        device = cluster_h.device

        # Sample random time
        t = torch.rand(batch_size, device=device)

        # Sample noisy state
        xt = self.diffusion.sample_forward(cluster_adj, t)

        # Predict clean state
        logits = self.coarse_score_net(cluster_h, xt, t)

        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.reshape(-1, 2),
            cluster_adj.long().reshape(-1),
        )

        return loss

    def fine_diffusion_loss(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        assignments: torch.Tensor,
        num_clusters: int,
    ) -> torch.Tensor:
        """
        Compute fine-level diffusion loss for intra-cluster routing.

        Args:
            h: Node embeddings (batch, n, d)
            adj: Ground truth adjacency (batch, n, n)
            assignments: Cluster assignments (batch, n)
            num_clusters: Number of clusters

        Returns:
            Loss scalar
        """
        batch_size, n, d = h.shape
        device = h.device

        total_loss = 0.0
        num_clusters_processed = 0

        for b in range(batch_size):
            for k in range(num_clusters):
                mask = (assignments[b] == k)
                cluster_size = mask.sum().item()

                if cluster_size < 2:
                    continue

                # Extract cluster subgraph
                cluster_indices = mask.nonzero(as_tuple=True)[0]
                cluster_h = h[b, cluster_indices].unsqueeze(0)  # (1, m, d)
                cluster_adj = adj[b][cluster_indices][:, cluster_indices].unsqueeze(0)  # (1, m, m)

                # Sample time
                t = torch.rand(1, device=device)

                # Sample noisy state
                xt = self.diffusion.sample_forward(cluster_adj, t)

                # Predict
                logits = self.fine_score_net(cluster_h, xt, t)

                # Loss
                loss = F.cross_entropy(
                    logits.reshape(-1, 2),
                    cluster_adj.long().reshape(-1),
                )
                total_loss += loss
                num_clusters_processed += 1

        if num_clusters_processed > 0:
            total_loss /= num_clusters_processed

        return total_loss

    def compute_loss(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        assignments: torch.Tensor,
        num_clusters: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hierarchical diffusion loss.

        Args:
            h: Node embeddings (batch, n, d)
            adj: Ground truth adjacency (batch, n, n)
            assignments: Cluster assignments (batch, n)
            num_clusters: Number of clusters

        Returns:
            Dictionary of losses
        """
        # Compute cluster-level structures
        cluster_h = self.compute_cluster_embeddings(h, assignments, num_clusters)
        cluster_adj = self.get_cluster_adjacency(adj, assignments, num_clusters)

        # Coarse loss
        coarse_loss = self.coarse_diffusion_loss(cluster_h, cluster_adj)

        # Fine loss
        fine_loss = self.fine_diffusion_loss(h, adj, assignments, num_clusters)

        return {
            'coarse': coarse_loss,
            'fine': fine_loss,
            'total': coarse_loss + fine_loss,
        }

    @torch.no_grad()
    def sample_coarse(
        self,
        cluster_h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample cluster-level adjacency using reverse diffusion.

        Args:
            cluster_h: Cluster embeddings (batch, k, d)

        Returns:
            Predicted cluster adjacency (batch, k, k)
        """
        batch_size, k, _ = cluster_h.shape
        device = cluster_h.device

        # Start from random
        xt = torch.randint(0, 2, (batch_size, k, k), device=device)
        xt = (xt + xt.transpose(1, 2)).clamp(0, 1)  # Symmetrize

        # Reverse diffusion
        timesteps = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, device=device)

        for i in range(self.num_inference_steps):
            t = timesteps[i].expand(batch_size)
            t_next = timesteps[i + 1].expand(batch_size)

            # Predict clean state
            logits = self.coarse_score_net(cluster_h, xt, t)
            probs = F.softmax(logits, dim=-1)[..., 1]

            # For final steps, use deterministic decoding
            if i >= self.num_inference_steps - 5:
                xt = (probs > 0.5).long()
            else:
                # Stochastic sampling
                xt = torch.bernoulli(probs).long()

            # Symmetrize
            xt = ((xt + xt.transpose(1, 2)) > 0).long()

        return xt

    @torch.no_grad()
    def sample_fine(
        self,
        h: torch.Tensor,
        assignments: torch.Tensor,
        num_clusters: int,
    ) -> List[torch.Tensor]:
        """
        Sample intra-cluster routing using reverse diffusion.

        Args:
            h: Node embeddings (batch, n, d)
            assignments: Cluster assignments (batch, n)
            num_clusters: Number of clusters

        Returns:
            List of cluster adjacency predictions
        """
        batch_size, n, d = h.shape
        device = h.device

        cluster_adjs = []

        for k in range(num_clusters):
            mask = (assignments[0] == k)  # Assume batch_size=1 for inference
            cluster_size = mask.sum().item()

            if cluster_size < 2:
                cluster_adjs.append(torch.zeros(cluster_size, cluster_size, device=device))
                continue

            cluster_indices = mask.nonzero(as_tuple=True)[0]
            cluster_h = h[0, cluster_indices].unsqueeze(0)

            # Start from random
            xt = torch.randint(0, 2, (1, cluster_size, cluster_size), device=device)
            xt = (xt + xt.transpose(1, 2)).clamp(0, 1)

            # Reverse diffusion
            timesteps = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, device=device)

            for i in range(self.num_inference_steps):
                t = timesteps[i].unsqueeze(0)

                logits = self.fine_score_net(cluster_h, xt, t)
                probs = F.softmax(logits, dim=-1)[..., 1]

                if i >= self.num_inference_steps - 5:
                    xt = (probs > 0.5).long()
                else:
                    xt = torch.bernoulli(probs).long()

                xt = ((xt + xt.transpose(1, 2)) > 0).long()

            cluster_adjs.append(xt[0])

        return cluster_adjs
