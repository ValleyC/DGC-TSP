"""
DGC-TSP: Main Model

Combines EGNN encoder, deep graph clustering, and hierarchical diffusion
for end-to-end TSP solving.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np

from .encoder import EGNNEncoder
from .clustering import DeepGraphClustering
from .diffusion import HierarchicalDiffusion


class DGCTSP(nn.Module):
    """
    DGC-TSP: Deep Graph Clustering for Hierarchical TSP.

    End-to-end model that:
    1. Encodes TSP instances with E(n)-equivariant GNN
    2. Learns tour-aware clustering
    3. Applies hierarchical diffusion for solution prediction
    """

    def __init__(
        self,
        coord_dim: int = 2,
        hidden_dim: int = 128,
        num_clusters: int = 10,
        num_egnn_layers: int = 4,
        num_diffusion_layers: int = 3,
        num_heads: int = 8,
        num_inference_steps: int = 50,
        lambda_clustering: float = 1.0,
        lambda_diffusion: float = 1.0,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.lambda_clustering = lambda_clustering
        self.lambda_diffusion = lambda_diffusion

        # E(n)-equivariant encoder
        self.encoder = EGNNEncoder(
            input_dim=coord_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_egnn_layers,
        )

        # Deep graph clustering
        self.clustering = DeepGraphClustering(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_clusters=num_clusters,
            temperature=temperature,
        )

        # Hierarchical diffusion
        self.diffusion = HierarchicalDiffusion(
            node_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            num_layers=num_diffusion_layers,
            num_heads=num_heads,
            num_inference_steps=num_inference_steps,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode node coordinates to embeddings.

        Args:
            x: Node coordinates (batch, n, 2)

        Returns:
            Node embeddings (batch, n, hidden_dim)
        """
        return self.encoder(x)

    def cluster(
        self,
        h: torch.Tensor,
        hard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cluster assignments.

        Args:
            h: Node embeddings (batch, n, d)
            hard: Whether to use hard assignments

        Returns:
            Clustering output dictionary
        """
        return self.clustering(h, hard=hard)

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        tour: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            x: Node coordinates (batch, n, 2)
            adj: Ground truth adjacency (batch, n, n) - for diffusion loss
            tour: Ground truth tour (batch, n) - for clustering alignment loss

        Returns:
            Dictionary containing losses and outputs
        """
        # Encode
        h = self.encode(x)

        # Clustering losses
        cluster_losses, cluster_output = self.clustering.compute_loss(h, tour)
        assignments = cluster_output['assignments'].argmax(dim=-1)

        outputs = {
            'embeddings': h,
            'cluster_assignments': assignments,
            'cluster_soft_assignments': cluster_output['assignments'],
        }

        # Diffusion losses (if adjacency provided)
        if adj is not None:
            diffusion_losses = self.diffusion.compute_loss(
                h, adj, assignments, self.num_clusters
            )
            outputs['diffusion_losses'] = diffusion_losses
        else:
            diffusion_losses = {'total': torch.tensor(0.0, device=x.device)}

        # Total loss
        total_loss = (
            self.lambda_clustering * cluster_losses['total'] +
            self.lambda_diffusion * diffusion_losses['total']
        )

        outputs['losses'] = {
            'clustering': cluster_losses,
            'diffusion': diffusion_losses,
            'total': total_loss,
        }

        return outputs

    @torch.no_grad()
    def solve(
        self,
        x: torch.Tensor,
        return_clusters: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Solve TSP instance using hierarchical diffusion.

        Args:
            x: Node coordinates (batch, n, 2) or (n, 2)
            return_clusters: Whether to return cluster information

        Returns:
            Dictionary containing predicted tour and optionally cluster info
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        batch_size, n, _ = x.shape
        device = x.device

        # Encode
        h = self.encode(x)

        # Get cluster assignments
        cluster_output = self.cluster(h, hard=True)
        assignments = cluster_output['assignments'].argmax(dim=-1)

        # Get cluster embeddings
        cluster_h = self.diffusion.compute_cluster_embeddings(
            h, assignments, self.num_clusters
        )

        # Sample coarse structure (cluster connections)
        cluster_adj = self.diffusion.sample_coarse(cluster_h)

        # Sample fine structure (intra-cluster routing)
        cluster_adjs = self.diffusion.sample_fine(h, assignments, self.num_clusters)

        # Assemble full solution
        full_adj = self.assemble_solution(
            cluster_adj[0], cluster_adjs, assignments[0], n, device
        )

        # Extract tour from adjacency
        tour = self.extract_tour(full_adj)

        outputs = {
            'tour': tour,
            'adjacency': full_adj,
        }

        if return_clusters:
            outputs['cluster_assignments'] = assignments
            outputs['cluster_adjacency'] = cluster_adj

        return outputs

    def assemble_solution(
        self,
        cluster_adj: torch.Tensor,
        cluster_adjs: List[torch.Tensor],
        assignments: torch.Tensor,
        n: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Assemble full solution from hierarchical predictions.

        Args:
            cluster_adj: Cluster-level adjacency (k, k)
            cluster_adjs: List of intra-cluster adjacencies
            assignments: Node cluster assignments (n,)
            n: Number of nodes
            device: Device

        Returns:
            Full node-level adjacency (n, n)
        """
        full_adj = torch.zeros(n, n, device=device)

        # Add intra-cluster edges
        for k, cadj in enumerate(cluster_adjs):
            mask = (assignments == k)
            indices = mask.nonzero(as_tuple=True)[0]

            for i, ni in enumerate(indices):
                for j, nj in enumerate(indices):
                    if cadj[i, j] > 0.5:
                        full_adj[ni, nj] = 1

        # Add inter-cluster edges based on cluster adjacency
        # Find boundary nodes for each cluster
        k = cluster_adj.shape[0]
        for ci in range(k):
            for cj in range(ci + 1, k):
                if cluster_adj[ci, cj] > 0.5:
                    # Find closest pair of nodes between clusters
                    mask_i = (assignments == ci)
                    mask_j = (assignments == cj)

                    if mask_i.sum() == 0 or mask_j.sum() == 0:
                        continue

                    # For now, just connect first available nodes
                    # In practice, should use distances
                    nodes_i = mask_i.nonzero(as_tuple=True)[0]
                    nodes_j = mask_j.nonzero(as_tuple=True)[0]

                    if len(nodes_i) > 0 and len(nodes_j) > 0:
                        # Simple: connect first nodes (should use heuristic)
                        ni, nj = nodes_i[0], nodes_j[0]
                        full_adj[ni, nj] = 1
                        full_adj[nj, ni] = 1

        return full_adj

    def extract_tour(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Extract tour from adjacency matrix using greedy approach.

        Args:
            adj: Adjacency matrix (n, n)

        Returns:
            Tour as node indices (n,)
        """
        n = adj.shape[0]
        adj_np = adj.cpu().numpy()

        # Find node degrees
        degrees = adj_np.sum(axis=1)

        # Start from node with odd degree (if any) or 0
        start = 0
        for i in range(n):
            if degrees[i] == 1:  # Path endpoint
                start = i
                break

        # Greedy traversal
        tour = [start]
        visited = {start}
        current = start

        while len(tour) < n:
            # Find unvisited neighbor
            neighbors = np.where(adj_np[current] > 0.5)[0]
            next_node = None

            for neighbor in neighbors:
                if neighbor not in visited:
                    next_node = neighbor
                    break

            if next_node is None:
                # No unvisited neighbor, find closest unvisited
                unvisited = [i for i in range(n) if i not in visited]
                if unvisited:
                    next_node = unvisited[0]
                else:
                    break

            tour.append(next_node)
            visited.add(next_node)
            current = next_node

        return torch.tensor(tour, device=adj.device)


class DGCTSPLightning:
    """
    PyTorch Lightning wrapper for DGC-TSP.

    Provides training and validation loops.
    """

    def __init__(self, model: DGCTSP, lr: float = 1e-4):
        self.model = model
        self.lr = lr

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Training step.

        Args:
            batch: Dictionary with 'coords', 'adj', 'tour'

        Returns:
            Loss tensor
        """
        outputs = self.model(
            x=batch['coords'],
            adj=batch['adj'],
            tour=batch['tour'],
        )
        return outputs['losses']['total']

    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Validation step.

        Args:
            batch: Dictionary with 'coords', 'adj', 'tour'

        Returns:
            Dictionary with metrics
        """
        # Compute losses
        outputs = self.model(
            x=batch['coords'],
            adj=batch['adj'],
            tour=batch['tour'],
        )

        # Solve and evaluate
        with torch.no_grad():
            solution = self.model.solve(batch['coords'])
            pred_tour = solution['tour']

        # Compute tour length
        coords = batch['coords'][0]  # Assume batch_size=1
        pred_length = self.compute_tour_length(coords, pred_tour)
        gt_length = self.compute_tour_length(coords, batch['tour'][0])

        gap = (pred_length - gt_length) / gt_length * 100

        return {
            'loss': outputs['losses']['total'],
            'pred_length': pred_length,
            'gt_length': gt_length,
            'gap': gap,
        }

    @staticmethod
    def compute_tour_length(coords: torch.Tensor, tour: torch.Tensor) -> float:
        """Compute tour length."""
        n = len(tour)
        length = 0.0
        for i in range(n):
            j = (i + 1) % n
            dist = torch.norm(coords[tour[i]] - coords[tour[j]]).item()
            length += dist
        return length
