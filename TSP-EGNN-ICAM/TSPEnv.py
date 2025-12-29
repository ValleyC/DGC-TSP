"""
TSP Environment for GEPNet.

Based on UDC's TSPEnv - handles SHPP (open path) sub-problems.
"""

import math
from dataclasses import dataclass
import torch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TSProblemDef import get_random_problems, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    problems: torch.Tensor = None
    dist: torch.Tensor = None
    log_scale: float = None


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None


class TSPEnv:
    """TSP Environment for training and evaluation."""

    def __init__(self, **env_params):
        self.env_params = env_params
        self.problem_size_low = env_params['problem_size_low']
        self.problem_size_high = env_params['problem_size_high']
        self.problem_size = env_params['sub_size']
        self.sample_size = env_params['sample_size']
        self.pomo_size = env_params['pomo_size']
        self.fs_sample_size = env_params.get('fs_sample_size', self.sample_size)

        # Problem data
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.problems = None
        self.raw_problems = None
        self.raw_problem_size = None

        # Dynamic state
        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None
        self.dist = None

        # States
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_raw_problems(self, batch_size, episode=1, nodes_coords=None):
        """Load or generate large TSP instance for partitioning."""
        if nodes_coords is not None:
            self.raw_problems = nodes_coords[episode:episode + batch_size]
        else:
            n_subs = np.random.randint(
                self.problem_size_low // self.problem_size,
                self.problem_size_high // self.problem_size + 1
            )
            self.raw_problem_size = n_subs * self.problem_size
            self.raw_problems = get_random_problems(batch_size, self.raw_problem_size)

    def load_problems(self, batch_size, subp, aug_factor=1):
        """Load sub-problems for POMO solving."""
        self.batch_size = batch_size
        self.problems = subp
        self.dist = torch.cdist(subp, subp, p=2)

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)
            else:
                raise NotImplementedError(f"Aug factor {aug_factor} not supported")

        device = self.problems.device
        self.BATCH_IDX = torch.arange(self.batch_size, device=device)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size, device=device)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        """Reset environment for new episode."""
        device = self.problems.device
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros(
            (self.batch_size, self.pomo_size, 0),
            dtype=torch.long,
            device=device
        )

        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros(
            (self.batch_size, self.pomo_size, self.problem_size),
            device=device
        )

        self.dist = (self.problems[:, :, None, :] - self.problems[:, None, :, :]).norm(p=2, dim=-1)
        log_scale = math.log2(self.problem_size)

        # POMO: half start from node 0, half from last node
        self.step_state.ninf_mask[:, :self.pomo_size // 2, -1] = float('-inf')
        self.step_state.ninf_mask[:, self.pomo_size // 2:, 0] = float('-inf')

        self.reset_state.problems = self.problems
        self.reset_state.dist = self.dist
        self.reset_state.log_scale = log_scale

        return self.reset_state, None, False

    def pre_step(self):
        """Return current state."""
        return self.step_state, None, False

    def step(self, selected):
        """Take a step."""
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat(
            (self.selected_node_list, self.current_node[:, :, None]),
            dim=2
        )

        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')

        # SHPP: n-1 steps for n nodes (open path)
        done = (self.selected_count >= self.problem_size - 1)
        if done and self.selected_node_list.size(-1) == self.problem_size:
            reward = -self._get_open_travel_distance()
        else:
            reward = None

        return self.step_state, reward, done

    def _get_open_travel_distance(self):
        """Compute SHPP (open path) distance."""
        gathering_index = self.selected_node_list.unsqueeze(3).expand(
            self.batch_size, -1, self.problem_size, 2
        )
        seq_expanded = self.problems[:, None, :, :].expand(
            self.batch_size, self.pomo_size, self.problem_size, 2
        )

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()

        # Open path: don't count last edge
        open_travel_distances = segment_lengths[:, :, :-1].sum(2)
        return open_travel_distances

    def get_open_travel_distance(self, problems, solution):
        """Compute SHPP distance for arbitrary problems and solutions."""
        batch_size = solution.size(0)
        pomo_size = solution.size(1)
        problem_size = solution.size(2)

        gathering_index = solution.unsqueeze(3).expand(batch_size, pomo_size, -1, 2)
        seq_expanded = problems[:, None, :, :].expand(batch_size, pomo_size, -1, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()

        travel_distances = segment_lengths[:, :, :-1].sum(2)
        return travel_distances

    def _get_travel_distance(self, problems, solution):
        """Compute closed TSP tour distance."""
        solution = solution[None, :]
        batch_size = solution.size(0)
        pomo_size = solution.size(1)

        gathering_index = solution.unsqueeze(3).expand(batch_size, pomo_size, -1, 2)
        seq_expanded = problems[:, None, :, :].expand(batch_size, pomo_size, -1, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()

        travel_distances = segment_lengths.sum(2)
        return travel_distances

    def _get_travel_distance2(self, problems, solution):
        """Compute closed TSP tour distance (batched)."""
        batch_size = solution.size(0)
        pomo_size = solution.size(1)

        gathering_index = solution.unsqueeze(3).expand(batch_size, pomo_size, -1, 2)
        seq_expanded = problems[:, None, :, :].expand(batch_size, pomo_size, -1, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()

        travel_distances = segment_lengths.sum(2)
        return travel_distances

    def get_local_feature(self):
        """Get distance from current node to all nodes."""
        if self.current_node is None:
            return None
        current_node = self.current_node[:, :, None, None].expand(
            self.batch_size, self.pomo_size, 1, self.problem_size
        )
        cur_dist = self.dist[:, None, :, :].expand(
            self.batch_size, self.pomo_size, self.problem_size, self.problem_size
        ).gather(2, current_node).squeeze(2)
        return cur_dist

    def make_dataset(self, filename, episode):
        """Load TSP instances from file."""
        nodes_coords = []
        tour = []

        for line in open(filename, "r").readlines()[0:episode]:
            line = line.split(" ")
            num_nodes = int(line.index('output') // 2)
            nodes_coords.append(
                [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]
            )
            tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]]
            tour.append(tour_nodes)

        return torch.tensor(nodes_coords), torch.tensor(tour)
