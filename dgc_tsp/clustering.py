"""
Deep Graph Clustering Module for TSP

This module learns to partition TSP graphs in a way that:
1. Minimizes inter-cluster edges in the optimal tour
2. Creates balanced clusters for parallel processing
3. Adapts to problem structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np


class SoftClusterAssignment(nn.Module):
    """
    Soft cluster assignment using attention mechanism.

    Produces differentiable soft assignments that can be used
    for end-to-end training with tour-aware losses.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_clusters: int = 10,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature

        # Learnable cluster centroids in embedding space
        self.cluster_centroids = nn.Parameter(
            torch.randn(num_clusters, input_dim) * 0.1
        )

        # MLP for transforming node embeddings before assignment
        self.transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Attention-based assignment
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)

    def forward(
        self,
        h: torch.Tensor,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cluster assignments.

        Args:
            h: Node embeddings (batch_size, n, input_dim)
            hard: If True, return hard assignments (argmax)

        Returns:
            assignments: Soft assignments (batch_size, n, num_clusters)
            cluster_centroids: Current cluster centroids
        """
        batch_size, n, d = h.shape

        # Transform embeddings
        h_transformed = self.transform(h)

        # Compute attention scores to cluster centroids
        queries = self.query_proj(h_transformed)  # (batch, n, d)
        keys = self.key_proj(self.cluster_centroids.unsqueeze(0).expand(batch_size, -1, -1))  # (batch, k, d)

        # Attention: (batch, n, d) @ (batch, d, k) -> (batch, n, k)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (d ** 0.5)
        scores = scores / self.temperature

        # Soft assignment via softmax
        assignments = F.softmax(scores, dim=-1)

        if hard:
            # Straight-through estimator for hard assignments
            hard_assignments = F.one_hot(assignments.argmax(dim=-1), self.num_clusters).float()
            assignments = hard_assignments - assignments.detach() + assignments

        return assignments, self.cluster_centroids


class DeepGraphClustering(nn.Module):
    """
    Deep Graph Clustering for TSP.

    Combines representation learning with clustering in a way
    that aligns with optimal tour structure.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_clusters: int = 10,
        temperature: float = 0.5,
        lambda_balance: float = 0.1,
        lambda_tour: float = 1.0,
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.lambda_balance = lambda_balance
        self.lambda_tour = lambda_tour

        # Soft cluster assignment module
        self.cluster_assignment = SoftClusterAssignment(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_clusters=num_clusters,
            temperature=temperature,
        )

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # L2 normalization for contrastive learning
        self.l2_norm = lambda x: F.normalize(x, p=2, dim=-1)

    def forward(
        self,
        h: torch.Tensor,
        hard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            h: Node embeddings from encoder (batch_size, n, input_dim)
            hard: Whether to use hard cluster assignments

        Returns:
            Dictionary containing:
                - assignments: Cluster assignments (batch, n, k)
                - projections: Projected embeddings for contrastive loss
                - centroids: Cluster centroids
        """
        # Get cluster assignments
        assignments, centroids = self.cluster_assignment(h, hard=hard)

        # Project for contrastive learning
        projections = self.projection_head(h)
        projections = self.l2_norm(projections)

        return {
            'assignments': assignments,
            'projections': projections,
            'centroids': centroids,
        }

    def contrastive_loss(
        self,
        projections: torch.Tensor,
        assignments: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute contrastive loss (InfoNCE-style).

        Nodes in the same cluster should have similar embeddings.

        Args:
            projections: L2-normalized projections (batch, n, d)
            assignments: Soft cluster assignments (batch, n, k)
            temperature: Temperature for contrastive loss

        Returns:
            Contrastive loss scalar
        """
        batch_size, n, d = projections.shape

        # Compute similarity matrix
        sim_matrix = torch.bmm(projections, projections.transpose(1, 2))  # (batch, n, n)
        sim_matrix = sim_matrix / temperature

        # Compute cluster membership similarity (how likely two nodes are in same cluster)
        cluster_sim = torch.bmm(assignments, assignments.transpose(1, 2))  # (batch, n, n)

        # Mask diagonal
        mask = torch.eye(n, device=projections.device).unsqueeze(0).expand(batch_size, -1, -1)
        sim_matrix = sim_matrix - mask * 1e9

        # InfoNCE loss: similar cluster members should have high similarity
        # Use cluster_sim as soft positive weights
        exp_sim = torch.exp(sim_matrix)
        positive_sim = (exp_sim * cluster_sim).sum(dim=-1)
        negative_sim = exp_sim.sum(dim=-1)

        loss = -torch.log(positive_sim / (negative_sim + 1e-8) + 1e-8)
        return loss.mean()

    def balance_loss(self, assignments: torch.Tensor) -> torch.Tensor:
        """
        Compute balance loss to ensure roughly equal cluster sizes.

        Args:
            assignments: Soft cluster assignments (batch, n, k)

        Returns:
            Balance loss scalar
        """
        # Average assignment per cluster
        cluster_sizes = assignments.mean(dim=1)  # (batch, k)

        # Target: uniform distribution
        target = torch.ones_like(cluster_sizes) / self.num_clusters

        # KL divergence from uniform
        loss = F.kl_div(
            torch.log(cluster_sizes + 1e-8),
            target,
            reduction='batchmean'
        )
        return loss

    def tour_alignment_loss(
        self,
        assignments: torch.Tensor,
        tour: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute tour alignment loss.

        Encourages cluster assignments to minimize edge cuts in the optimal tour.
        Adjacent nodes in tour should preferably be in the same cluster.

        Args:
            assignments: Soft cluster assignments (batch, n, k)
            tour: Optimal tour indices (batch, n) or adjacency (batch, n, n)

        Returns:
            Tour alignment loss scalar
        """
        batch_size, n, k = assignments.shape

        if tour.dim() == 2:
            # Convert tour sequence to adjacency
            tour_adj = torch.zeros(batch_size, n, n, device=assignments.device)
            for b in range(batch_size):
                for i in range(n):
                    j = (i + 1) % n
                    tour_adj[b, tour[b, i], tour[b, j]] = 1
                    tour_adj[b, tour[b, j], tour[b, i]] = 1
        else:
            tour_adj = tour

        # Compute cluster membership similarity
        cluster_sim = torch.bmm(assignments, assignments.transpose(1, 2))  # (batch, n, n)

        # Loss: edges in tour should have high cluster similarity
        # (1 - cluster_sim) * tour_adj counts "cut" edges
        cut_penalty = (1 - cluster_sim) * tour_adj
        loss = cut_penalty.sum(dim=(1, 2)) / (tour_adj.sum(dim=(1, 2)) + 1e-8)

        return loss.mean()

    def compute_loss(
        self,
        h: torch.Tensor,
        tour: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all clustering losses.

        Args:
            h: Node embeddings (batch, n, d)
            tour: Optional optimal tour for tour alignment loss

        Returns:
            Dictionary of loss components and total loss
        """
        output = self.forward(h)

        losses = {}

        # Contrastive loss
        losses['contrastive'] = self.contrastive_loss(
            output['projections'],
            output['assignments'],
        )

        # Balance loss
        losses['balance'] = self.balance_loss(output['assignments'])

        # Tour alignment loss (if tour provided)
        if tour is not None:
            losses['tour_alignment'] = self.tour_alignment_loss(
                output['assignments'],
                tour,
            )
        else:
            losses['tour_alignment'] = torch.tensor(0.0, device=h.device)

        # Total loss
        losses['total'] = (
            losses['contrastive'] +
            self.lambda_balance * losses['balance'] +
            self.lambda_tour * losses['tour_alignment']
        )

        return losses, output

    def get_hard_assignments(self, h: torch.Tensor) -> torch.Tensor:
        """
        Get hard cluster assignments.

        Args:
            h: Node embeddings (batch, n, d)

        Returns:
            Hard cluster assignments (batch, n)
        """
        output = self.forward(h, hard=True)
        return output['assignments'].argmax(dim=-1)

    def get_cluster_nodes(
        self,
        assignments: torch.Tensor,
    ) -> list:
        """
        Get list of node indices for each cluster.

        Args:
            assignments: Hard assignments (n,) for single graph

        Returns:
            List of lists containing node indices for each cluster
        """
        clusters = []
        for k in range(self.num_clusters):
            nodes = (assignments == k).nonzero(as_tuple=True)[0].tolist()
            clusters.append(nodes)
        return clusters


class KMeansClusteringBaseline(nn.Module):
    """
    K-Means clustering baseline (non-learned).

    Uses coordinate-based K-Means for comparison.
    """

    def __init__(self, num_clusters: int = 10, num_iters: int = 10):
        super().__init__()
        self.num_clusters = num_clusters
        self.num_iters = num_iters

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform K-Means clustering on coordinates.

        Args:
            x: Node coordinates (batch, n, 2)

        Returns:
            Cluster assignments (batch, n)
        """
        batch_size, n, d = x.shape
        assignments = torch.zeros(batch_size, n, dtype=torch.long, device=x.device)

        for b in range(batch_size):
            points = x[b]  # (n, d)

            # Initialize centroids randomly
            indices = torch.randperm(n)[:self.num_clusters]
            centroids = points[indices].clone()

            for _ in range(self.num_iters):
                # Assign points to nearest centroid
                dists = torch.cdist(points, centroids)  # (n, k)
                cluster_ids = dists.argmin(dim=1)

                # Update centroids
                for k in range(self.num_clusters):
                    mask = (cluster_ids == k)
                    if mask.sum() > 0:
                        centroids[k] = points[mask].mean(dim=0)

            assignments[b] = cluster_ids

        return assignments
