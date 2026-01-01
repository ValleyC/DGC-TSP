"""
EDGC Trainer for CVRP.

One-shot clustering approach replacing autoregressive partition.

Key differences from CVRPTrainerPartition:
1. O(N) one-shot clustering instead of O(N^2) autoregressive
2. Combined loss: InfoNCE + KL + Capacity + REINFORCE
3. Cluster-to-route conversion with capacity splitting
"""

import torch
import numpy as np
from logging import getLogger
from torch_geometric.data import Data
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model
from ClusterPartitionModel import ClusterPartitionModel
from ClusterLosses import EDGCLoss, compute_cluster_statistics
from utils.utils import get_result_folder, AverageMeter, LogData, TimeEstimator, util_print_log_array


class EDGCTrainer:
    """
    EDGC Trainer for CVRP.

    Uses one-shot clustering for partition instead of autoregressive sampling.
    """

    def __init__(self, env_params, model_params, model_p_params, optimizer_params, trainer_params):
        self.env_params = env_params
        self.model_params = model_params
        self.model_p_params = model_p_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # CUDA setup
        USE_CUDA = trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Cluster Partition Model (EDGC)
        self.model_p = ClusterPartitionModel(
            units=model_p_params['embedding_dim'],
            node_feats=2,  # (demand, r) - E(2)-invariant
            edge_feats=2,
            depth=model_p_params['depth'],
            projection_dim=model_p_params.get('projection_dim', 128),
            max_clusters=model_p_params.get('max_clusters', 50)
        ).to(self.device)

        # Sub-problem solver (TSP model)
        self.model_t = Model(**model_params)
        self.env = Env(**env_params)

        # EDGC Loss
        self.edgc_loss = EDGCLoss(
            w_infonce=trainer_params.get('w_infonce', 1.0),
            w_kl=trainer_params.get('w_kl', 0.1),
            w_capacity=trainer_params.get('w_capacity', 10.0),
            w_balance=trainer_params.get('w_balance', 0.01),
            w_reinforce=trainer_params.get('w_reinforce', 1.0),
            temperature=trainer_params.get('temperature', 0.5)
        )

        # Optimizers
        self.optimizer_p = Optimizer(self.model_p.parameters(), **optimizer_params['optimizer_p'])
        self.optimizer_t = Optimizer(self.model_t.parameters(), **optimizer_params['optimizer'])
        self.scheduler_p = Scheduler(self.optimizer_p, **optimizer_params['scheduler'])
        self.scheduler_t = Scheduler(self.optimizer_t, **optimizer_params['scheduler'])

        self.start_epoch = 1
        self._load_checkpoints()
        self.time_estimator = TimeEstimator()

    def _load_checkpoints(self):
        """Load pre-trained models if specified."""
        model_load = self.trainer_params['model_load']

        if model_load['t_enable']:
            checkpoint_fullname = '{t_path}/checkpoint-tsp-{t_epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.model_t.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['t_epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer_t.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler_t.last_epoch = model_load['t_epoch'] - 1
            self.logger.info('Loaded TSP Model!')

        if model_load['p_enable']:
            checkpoint_fullname = '{p_path}/checkpoint-partition-{p_epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.model_p.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['p_epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer_p.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler_p.last_epoch = model_load['p_epoch'] - 1
            self.logger.info('Loaded EDGC Partition Model!')

    def run(self):
        """Main training loop."""
        self.time_estimator.reset(self.start_epoch)

        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            self.logger.info('=' * 60)

            self.scheduler_p.step()
            self.scheduler_t.step()

            train_score, train_loss, cluster_stats = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))
            self.logger.info("Cluster Stats: {}".format(cluster_stats))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']

            if all_done or (epoch % model_save_interval) == 0:
                self._save_checkpoints(epoch)

            if all_done:
                self.logger.info(" *** Training Done *** ")
                util_print_log_array(self.logger, self.result_log)

    def _save_checkpoints(self, epoch):
        """Save model checkpoints."""
        self.logger.info("Saving checkpoints...")

        checkpoint_dict_t = {
            'epoch': epoch,
            'model_state_dict': self.model_t.state_dict(),
            'optimizer_state_dict': self.optimizer_t.state_dict(),
            'scheduler_state_dict': self.scheduler_t.state_dict(),
            'result_log': self.result_log.get_raw_data()
        }
        torch.save(checkpoint_dict_t, f'{self.result_folder}/checkpoint-tsp-{epoch}.pt')

        checkpoint_dict_p = {
            'epoch': epoch,
            'model_state_dict': self.model_p.state_dict(),
            'optimizer_state_dict': self.optimizer_p.state_dict(),
            'scheduler_state_dict': self.scheduler_p.state_dict(),
            'result_log': self.result_log.get_raw_data()
        }
        torch.save(checkpoint_dict_p, f'{self.result_folder}/checkpoint-partition-{epoch}.pt')

    def _train_one_epoch(self, epoch):
        """Train for one epoch."""
        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        cluster_stats_AM = {}

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0

        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss, stats = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            # Update cluster statistics
            for k, v in stats.items():
                if k not in cluster_stats_AM:
                    cluster_stats_AM[k] = AverageMeter()
                cluster_stats_AM[k].update(v, batch_size)

            episode += batch_size

            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info(
                        'Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                        .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                score_AM.avg, loss_AM.avg))

        self.logger.info(
            'Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
            .format(epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg))

        final_stats = {k: v.avg for k, v in cluster_stats_AM.items()}
        return score_AM.avg, loss_AM.avg, final_stats

    def _train_one_batch(self, batch_size):
        """Train on one batch using one-shot clustering."""
        self.model_t.train()
        self.model_p.train()

        # Generate problems
        self.env.load_raw_problems(batch_size)
        pyg_data = self.gen_pyg_data(self.env.raw_depot_node_xy, self.env.raw_depot_node_demand)

        # Get demands (excluding depot)
        demands = self.env.raw_depot_node_demand[0, 1:]  # (n_nodes-1,)

        # Estimate number of clusters based on total demand
        total_demand = demands.sum()
        n_clusters = max(int(torch.ceil(total_demand).item()) + 1, 2)
        n_clusters = min(n_clusters, 50)

        # One-shot clustering (O(N) instead of O(N^2))
        q, z1, z2, k_logits, distances = self.model_p(
            pyg_data,
            n_clusters=n_clusters,
            is_train=True,
            sigma=0.01
        )

        # Get hard cluster assignments (for solver)
        # Use sampling for exploration
        labels, log_probs_cluster = self.model_p.get_hard_assignment(q[1:], sample=True)  # Exclude depot

        # Compute cluster statistics
        stats = compute_cluster_statistics(q[1:], demands)

        # Convert clusters to routes
        routes = self._cluster_to_routes(labels, self.env.raw_depot_node_xy[0], demands)

        # Solve sub-problems (TSP for each route)
        total_reward, loss_t = self._solve_routes(routes)

        # Compute EDGC loss
        total_loss, loss_dict = self.edgc_loss(
            z1[1:], z2[1:], q[1:],  # Exclude depot
            demands=demands,
            capacity=1.0,
            log_probs=log_probs_cluster.sum().unsqueeze(0),  # Sum log probs for REINFORCE
            rewards=total_reward.unsqueeze(0),
            baseline=None
        )

        # Add solver loss
        total_loss = total_loss + loss_t

        # Backward pass
        self.model_p.zero_grad()
        self.model_t.zero_grad()
        total_loss.backward()
        self.optimizer_p.step()
        self.optimizer_t.step()

        score = -total_reward.item()
        return score, total_loss.item(), stats

    def _cluster_to_routes(self, labels, coords, demands, capacity=1.0):
        """
        Convert cluster labels to CVRP routes.

        Args:
            labels: Cluster assignments (n_nodes,) - excludes depot
            coords: Node coordinates (n_nodes+1, 2) - includes depot
            demands: Node demands (n_nodes,) - excludes depot
            capacity: Vehicle capacity

        Returns:
            routes: List of routes, each is list of node indices (1-indexed)
        """
        device = labels.device
        unique_clusters = torch.unique(labels)
        routes = []
        depot_coord = coords[0]

        for cluster_id in unique_clusters:
            mask = labels == cluster_id
            cluster_nodes = torch.where(mask)[0] + 1  # +1 for depot offset
            cluster_demands = demands[mask]

            # Sort by angle from depot
            cluster_coords = coords[cluster_nodes]
            rel_coords = cluster_coords - depot_coord
            angles = torch.atan2(rel_coords[:, 1], rel_coords[:, 0])
            sorted_indices = torch.argsort(angles)

            # Split into routes based on capacity
            current_route = []
            current_demand = 0.0

            for idx in sorted_indices:
                node = cluster_nodes[idx].item()
                demand = cluster_demands[idx].item()

                if current_demand + demand > capacity + 1e-6:
                    if current_route:
                        routes.append(current_route)
                    current_route = [node]
                    current_demand = demand
                else:
                    current_route.append(node)
                    current_demand += demand

            if current_route:
                routes.append(current_route)

        return routes

    def _solve_routes(self, routes):
        """
        Solve TSP for each route and return total reward.

        Args:
            routes: List of routes

        Returns:
            total_reward: Negative total tour length
            loss_t: Solver training loss
        """
        total_length = 0.0
        loss_t_total = 0.0

        for route in routes:
            if len(route) <= 1:
                # Single node or empty route
                if len(route) == 1:
                    # Just depot -> node -> depot distance
                    node_idx = route[0]
                    depot = self.env.raw_depot_node_xy[0, 0]
                    node = self.env.raw_depot_node_xy[0, node_idx]
                    dist = torch.norm(node - depot) * 2
                    total_length += dist.item()
                continue

            # Create sub-problem
            route_tensor = torch.tensor(route, device=self.device)
            tsp_coords = self.env.raw_depot_node_xy[0, route_tensor]
            depot = self.env.raw_depot_node_xy[0, 0:1]

            # Compute tour length using nearest neighbor heuristic
            # (For simplicity; in full implementation, use model_t)
            tour_length = self._compute_tour_length(depot, tsp_coords)
            total_length += tour_length

        total_reward = -torch.tensor(total_length, device=self.device)
        return total_reward, torch.tensor(loss_t_total, device=self.device)

    def _compute_tour_length(self, depot, nodes):
        """Compute simple tour length: depot -> nodes in order -> depot."""
        if nodes.shape[0] == 0:
            return 0.0

        coords = torch.cat([depot, nodes, depot], dim=0)
        diffs = coords[1:] - coords[:-1]
        lengths = torch.norm(diffs, dim=-1)
        return lengths.sum().item()

    def gen_pyg_data(self, coors, demand, k_sparse=100):
        """Generate PyG data with E(2)-invariant features."""
        n_nodes = demand.size(1)
        norm_demand = demand

        shift_coors = coors - coors[:, 0:1, :]
        _x, _y = shift_coors[:, :, 0], shift_coors[:, :, 1]
        r = torch.sqrt(_x ** 2 + _y ** 2)

        x = torch.stack((norm_demand, r)).permute(1, 2, 0)
        cos_mat = self.gen_cos_sim_matrix(shift_coors)

        topk_values, topk_indices = torch.topk(cos_mat, k=min(k_sparse, n_nodes), dim=2, largest=True)

        edge_index = torch.cat((
            torch.repeat_interleave(torch.arange(n_nodes, device=coors.device), repeats=min(k_sparse, n_nodes))[None, :],
            topk_indices.view(1, -1)
        ), dim=0)

        edge_attr1 = topk_values.reshape(1, -1, 1)
        edge_attr2 = cos_mat[0, edge_index[0], edge_index[1]].reshape(1, -1, 1)
        edge_attr = torch.cat((edge_attr1, edge_attr2), dim=2)

        pyg_data = Data(
            x=x[0],
            edge_index=edge_index,
            edge_attr=edge_attr[0],
            pos=coors[0]
        )
        return pyg_data

    def gen_cos_sim_matrix(self, shift_coors):
        """Compute cosine similarity matrix."""
        dot_products = torch.bmm(shift_coors, shift_coors.transpose(1, 2))
        magnitudes = torch.sqrt(torch.sum(shift_coors ** 2, dim=-1)).unsqueeze(-1)
        magnitude_matrix = torch.bmm(magnitudes, magnitudes.transpose(1, 2)) + 1e-10
        cosine_similarity_matrix = dot_products / magnitude_matrix
        return cosine_similarity_matrix
