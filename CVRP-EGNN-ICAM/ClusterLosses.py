"""
EDGC Loss Functions for CVRP Clustering.

Combines multiple loss components:
1. InfoNCE: Contrastive loss on dual projections (from RGC)
2. KL Clustering: Self-training with sharpened target (from SDCN)
3. REINFORCE: Solver reward for routing quality
4. Capacity: Penalty for capacity violations
5. Balance: Encourages balanced cluster sizes

References:
- RGC train.py lines 162-198: InfoNCE and KL implementation
- SDCN sdcn.py lines 154-158: Combined loss
- SCGC train.py: MSE reconstruction loss
"""

import torch
import torch.nn.functional as F


def info_nce_loss(z1, z2, temperature=0.5):
    """
    InfoNCE contrastive loss (from RGC).

    Pulls together two views of the same node while pushing apart
    views of different nodes.

    Reference: RGC train.py lines 162-174

    Args:
        z1: First view embeddings (n_nodes, dim), L2 normalized
        z2: Second view embeddings (n_nodes, dim), L2 normalized
        temperature: Softmax temperature

    Returns:
        loss: InfoNCE loss scalar
    """
    n_nodes = z1.shape[0]
    device = z1.device

    # Concatenate both views
    z = torch.cat([z1, z2], dim=0)  # (2*n_nodes, dim)

    # Similarity matrix
    sim = torch.mm(z, z.T) / temperature  # (2*n_nodes, 2*n_nodes)

    # Mask out self-similarity
    mask = torch.eye(2 * n_nodes, device=device).bool()
    sim.masked_fill_(mask, float('-inf'))

    # Positive pairs: (i, i+n) and (i+n, i)
    pos_mask = torch.zeros(2 * n_nodes, 2 * n_nodes, device=device).bool()
    pos_mask[torch.arange(n_nodes), torch.arange(n_nodes) + n_nodes] = True
    pos_mask[torch.arange(n_nodes) + n_nodes, torch.arange(n_nodes)] = True

    # InfoNCE: -log(exp(pos) / sum(exp(all)))
    pos_sim = sim[pos_mask].view(2 * n_nodes)
    neg_sim = sim.masked_fill(pos_mask, float('-inf'))

    # Compute log-softmax
    all_sim = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    log_prob = F.log_softmax(all_sim, dim=1)[:, 0]

    loss = -log_prob.mean()
    return loss


def kl_clustering_loss(q, p=None):
    """
    KL divergence clustering loss for self-training (from SDCN/RGC).

    If p is not provided, compute target distribution from q.

    Reference: RGC train.py lines 192-198, SDCN sdcn.py lines 107-109

    Args:
        q: Soft cluster assignment (n_nodes, n_clusters)
        p: Target distribution (optional)

    Returns:
        loss: KL divergence loss
    """
    if p is None:
        # Compute target distribution: p = q^2 / sum(q) / Z
        weight = q ** 2 / (q.sum(dim=0, keepdim=True) + 1e-8)
        p = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
        p = p.detach()

    loss = F.kl_div(torch.log(q + 1e-8), p, reduction='batchmean')
    return loss


def capacity_constraint_loss(q, demands, capacity=1.0, penalty_weight=10.0):
    """
    Soft penalty for capacity constraint violations.

    For each cluster k: sum_i(q_ik * d_i) <= capacity

    Args:
        q: Soft cluster assignment (n_nodes, n_clusters)
        demands: Node demands (n_nodes,)
        capacity: Vehicle capacity
        penalty_weight: Weight for violation penalty

    Returns:
        loss: Capacity violation penalty
    """
    # Compute expected demand per cluster
    cluster_demands = (q * demands.unsqueeze(1)).sum(dim=0)  # (n_clusters,)

    # Soft penalty for exceeding capacity
    violations = F.relu(cluster_demands - capacity)
    loss = penalty_weight * (violations ** 2).mean()

    return loss


def balance_loss(q, target_ratio=None):
    """
    Encourages balanced cluster sizes.

    Minimizes variance of cluster sizes or targets specific ratio.

    Args:
        q: Soft cluster assignment (n_nodes, n_clusters)
        target_ratio: Target cluster size ratio (optional)

    Returns:
        loss: Balance loss
    """
    cluster_sizes = q.sum(dim=0)  # (n_clusters,)
    n_nodes = q.shape[0]
    n_clusters = q.shape[1]

    if target_ratio is None:
        # Target uniform distribution
        target_size = n_nodes / n_clusters
        loss = ((cluster_sizes - target_size) ** 2).mean() / (n_nodes ** 2)
    else:
        target_sizes = target_ratio * n_nodes
        loss = ((cluster_sizes - target_sizes) ** 2).mean() / (n_nodes ** 2)

    return loss


def reinforce_partition_loss(log_probs, rewards, baseline=None):
    """
    REINFORCE loss for partition decisions.

    Used to train clustering based on downstream solver reward.

    Args:
        log_probs: Log probabilities of cluster assignments (n_samples,)
        rewards: Solver rewards (negative tour length) (n_samples,)
        baseline: Baseline for variance reduction (optional)

    Returns:
        loss: REINFORCE loss
    """
    if baseline is None:
        baseline = rewards.mean()

    advantage = rewards - baseline
    loss = -(advantage.detach() * log_probs).mean()

    return loss


def reconstruction_loss(z1, z2, adj_target):
    """
    Graph reconstruction loss (from SCGC).

    Reconstructs adjacency from embeddings: S = z1 @ z2.T

    Reference: SCGC train.py lines 105-106

    Args:
        z1: First view embeddings (n_nodes, dim)
        z2: Second view embeddings (n_nodes, dim)
        adj_target: Target adjacency matrix (n_nodes, n_nodes)

    Returns:
        loss: MSE reconstruction loss
    """
    S = torch.mm(z1, z2.T)
    loss = F.mse_loss(S, adj_target)
    return loss


class EDGCLoss(torch.nn.Module):
    """
    Combined EDGC loss for CVRP clustering.

    Total loss = w1*InfoNCE + w2*KL + w3*Capacity + w4*Balance + w5*REINFORCE

    The REINFORCE component is the primary signal for routing quality,
    while clustering losses provide auxiliary regularization.
    """

    def __init__(
        self,
        w_infonce=1.0,
        w_kl=0.1,
        w_capacity=10.0,
        w_balance=0.01,
        w_reinforce=1.0,
        temperature=0.5
    ):
        """
        Args:
            w_infonce: Weight for InfoNCE contrastive loss
            w_kl: Weight for KL clustering loss
            w_capacity: Weight for capacity constraint loss
            w_balance: Weight for balance loss
            w_reinforce: Weight for REINFORCE loss
            temperature: Temperature for InfoNCE
        """
        super().__init__()
        self.w_infonce = w_infonce
        self.w_kl = w_kl
        self.w_capacity = w_capacity
        self.w_balance = w_balance
        self.w_reinforce = w_reinforce
        self.temperature = temperature

    def forward(
        self,
        z1,
        z2,
        q,
        demands=None,
        capacity=1.0,
        log_probs=None,
        rewards=None,
        baseline=None
    ):
        """
        Compute combined EDGC loss.

        Args:
            z1: First view embeddings (n_nodes, dim)
            z2: Second view embeddings (n_nodes, dim)
            q: Soft cluster assignment (n_nodes, n_clusters)
            demands: Node demands (optional)
            capacity: Vehicle capacity
            log_probs: Log probs for REINFORCE (optional)
            rewards: Solver rewards (optional)
            baseline: REINFORCE baseline (optional)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        losses = {}

        # InfoNCE contrastive loss
        if self.w_infonce > 0:
            losses['infonce'] = info_nce_loss(z1, z2, self.temperature)
        else:
            losses['infonce'] = torch.tensor(0.0, device=z1.device)

        # KL clustering loss
        if self.w_kl > 0:
            losses['kl'] = kl_clustering_loss(q)
        else:
            losses['kl'] = torch.tensor(0.0, device=z1.device)

        # Capacity constraint loss
        if self.w_capacity > 0 and demands is not None:
            losses['capacity'] = capacity_constraint_loss(q, demands, capacity)
        else:
            losses['capacity'] = torch.tensor(0.0, device=z1.device)

        # Balance loss
        if self.w_balance > 0:
            losses['balance'] = balance_loss(q)
        else:
            losses['balance'] = torch.tensor(0.0, device=z1.device)

        # REINFORCE loss (primary signal)
        if self.w_reinforce > 0 and log_probs is not None and rewards is not None:
            losses['reinforce'] = reinforce_partition_loss(log_probs, rewards, baseline)
        else:
            losses['reinforce'] = torch.tensor(0.0, device=z1.device)

        # Combine losses
        total_loss = (
            self.w_infonce * losses['infonce'] +
            self.w_kl * losses['kl'] +
            self.w_capacity * losses['capacity'] +
            self.w_balance * losses['balance'] +
            self.w_reinforce * losses['reinforce']
        )

        return total_loss, losses


def compute_cluster_statistics(q, demands):
    """
    Compute cluster statistics for monitoring.

    Args:
        q: Soft cluster assignment (n_nodes, n_clusters)
        demands: Node demands (n_nodes,)

    Returns:
        stats: Dictionary of statistics
    """
    cluster_sizes = q.sum(dim=0)
    cluster_demands = (q * demands.unsqueeze(1)).sum(dim=0)

    stats = {
        'n_clusters_active': (cluster_sizes > 0.5).sum().item(),
        'cluster_size_mean': cluster_sizes.mean().item(),
        'cluster_size_std': cluster_sizes.std().item(),
        'cluster_demand_mean': cluster_demands.mean().item(),
        'cluster_demand_max': cluster_demands.max().item(),
        'assignment_entropy': -(q * torch.log(q + 1e-8)).sum(dim=1).mean().item()
    }

    return stats
