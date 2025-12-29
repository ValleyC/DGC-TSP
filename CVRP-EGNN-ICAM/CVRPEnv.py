"""
CVRP Environment for GEPNet.

Based on UDC's CVRPEnv with the same interface for compatibility.
"""

import math
import pickle
from dataclasses import dataclass
import torch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CVRProblemDef import get_random_problems, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    node_xy: torch.Tensor = None
    node_demand: torch.Tensor = None
    dist: torch.Tensor = None
    log_scale: float = None
    flag_return: torch.Tensor = None


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    selected_count: int = None
    load: torch.Tensor = None
    left: torch.Tensor = None
    solution_flag: torch.Tensor = None
    solution_list: torch.Tensor = None
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None
    finished: torch.Tensor = None


class CVRPEnv:
    """CVRP Environment for training and evaluation."""

    def __init__(self, **env_params):
        self.env_params = env_params
        self.problem_size_low = env_params['problem_size_low']
        self.problem_size_high = env_params['problem_size_high']
        self.problem_size = env_params['sub_size']
        self.sample_size = env_params['sample_size']
        self.pomo_size = env_params['pomo_size']

        # Problem data
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.depot_node_xy = None
        self.depot_node_demand = None

        # Dynamic state
        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None
        self.solution_list = None
        self.solution_flag = None
        self.node_count = None

        # Capacity tracking
        self.at_the_depot = None
        self.load = None
        self.visited_ninf_flag = None
        self.ninf_mask = None
        self.finished = None
        self.demand_last = None
        self.dist = None

        # States
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_raw_problems(self, batch_size, episode=1, nodes_coords=None, nodes_demands=None):
        """Load or generate large CVRP instance for partitioning."""
        if nodes_coords is not None:
            self.raw_depot_node_xy = nodes_coords[episode:episode + batch_size]
            self.raw_depot_node_demand = nodes_demands[episode:episode + batch_size]
            self.raw_problems = torch.cat((self.raw_depot_node_xy, self.raw_depot_node_demand[:, :, None]), dim=-1)
        else:
            self.raw_problem_size = np.random.randint(
                self.problem_size_low // self.problem_size,
                self.problem_size_high // self.problem_size + 1
            ) * self.problem_size
            depot_xy, node_xy, node_demand = get_random_problems(batch_size, self.raw_problem_size)
            self.raw_depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
            depot_demand = torch.zeros(size=(batch_size, 1))
            self.raw_depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
            self.raw_problems = torch.cat((self.raw_depot_node_xy, self.raw_depot_node_demand[:, :, None]), dim=-1)

    def load_problems(self, batch_size, depot_xy, node_xy, node_demand, flag, aug_factor=1):
        """Load sub-problems for solving."""
        self.batch_size = batch_size
        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        self.dist = torch.cdist(self.depot_node_xy, self.depot_node_xy, p=2)
        depot_demand = torch.zeros(size=(self.batch_size, 1), device=depot_xy.device)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        self.BATCH_IDX = torch.arange(self.batch_size, device=depot_xy.device)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size, device=depot_xy.device)[None, :].expand(self.batch_size, self.pomo_size)
        self.flag = flag
        self.demand_last = node_demand[:, -1].clone()
        self.demand_last[self.flag.bool()] = 0

        self.reset_state.flag_return = flag
        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self, capacity_now, capacity_end):
        """Reset environment for new episode."""
        device = self.depot_node_xy.device
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long, device=device)
        self.solution_list = torch.zeros((self.batch_size, self.pomo_size, self.problem_size + 1), dtype=torch.long, device=device)
        self.solution_flag = torch.zeros((self.batch_size, self.pomo_size, self.problem_size + 1), dtype=torch.long, device=device)
        self.node_count = -1 * torch.ones((self.batch_size, self.pomo_size, 1), dtype=torch.long, device=device)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool, device=device)
        first = torch.ones(size=(self.batch_size, self.pomo_size), device=device) * capacity_now
        last = torch.ones(size=(self.batch_size, self.pomo_size), device=device) * capacity_end
        self.load = first
        self.left = last
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1), device=device)
        self.last_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1), device=device)
        self.last_mask[:, :, -1][(self.flag == 0)[:, None].expand(-1, self.pomo_size)] = float('-inf')
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1), device=device)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool, device=device)
        self.dist = (self.depot_node_xy[:, :, None, :] - self.depot_node_xy[:, None, :, :]).norm(p=2, dim=-1)
        self.reset_state.dist = self.dist
        self.reset_state.log_scale = math.log2(self.problem_size)

        return self.reset_state, None, False

    def pre_step(self):
        """Return current state."""
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.left = self.left
        self.step_state.solution_flag = torch.zeros((self.batch_size, self.pomo_size, self.problem_size), dtype=torch.long, device=self.depot_node_xy.device)
        self.step_state.solution_list = torch.zeros((self.batch_size, self.pomo_size, self.problem_size), dtype=torch.long, device=self.depot_node_xy.device)
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        return self.step_state, None, False

    def step(self, selected):
        """Take a step in the environment."""
        device = selected.device
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)

        if self.selected_count > 1:
            self.solution_flag = self.solution_flag.scatter_add(dim=-1, index=self.node_count, src=(selected[:, :, None] == 0).long())
        self.node_count[selected[:, :, None] != 0] += 1
        self.solution_list = self.solution_list.scatter_add(dim=-1, index=self.node_count, src=selected[:, :, None])

        self.at_the_depot = (selected == 0)
        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        gathering_index = selected[:, :, None]
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        round_error_epsilon = 0.0001
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0

        self.ninf_mask = self.visited_ninf_flag.clone()
        condition_mask = ((self.visited_ninf_flag[:, :, 1:] == float('-inf')).sum(-1) < self.problem_size - 1) | (
                1 - self.load + self.demand_last[:, None] > self.left + round_error_epsilon)
        self.ninf_mask[condition_mask[:, :, None].expand_as(self.ninf_mask)] += self.last_mask[condition_mask[:, :, None].expand_as(self.ninf_mask)]
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        self.ninf_mask[demand_too_large] = float('-inf')

        newly_finished = (self.visited_ninf_flag[:, :, 1:] == float('-inf')).all(dim=2) & (1 - self.load < self.left + round_error_epsilon) & ~self.finished
        self.step_state.solution_list[newly_finished] = self.solution_list[:, :, :-1][newly_finished]
        self.step_state.solution_flag[newly_finished] = self.solution_flag[:, :, :-1][newly_finished]
        self.finished = self.finished + newly_finished

        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        done = self.finished.all()
        return self.step_state, None, done

    def cal_open_length(self, problems, order_node, order_flag):
        """Calculate open tour length (without return to depot at end)."""
        order_node_ = order_node.clone()
        order_flag_ = order_flag.clone()
        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)
        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = 0
        solution = torch.stack((order_node_, order_flag_), dim=3).view(order_node_.size(0), order_node_.size(1), -1)
        batch_size = solution.size(0)
        pomo_size = solution.size(1)
        gathering_index = solution.unsqueeze(3).expand(batch_size, pomo_size, -1, 2)
        seq_expanded = problems[:, None, :, :].expand(batch_size, pomo_size, -1, 2)
        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        travel_distances = segment_lengths[:, :, :-1].sum(2)
        return travel_distances

    def cal_length(self, problems, order_node, order_flag):
        """Calculate closed tour length."""
        return self.cal_open_length(problems, order_node, order_flag)

    def cal_length_total(self, problems, order_node, order_flag):
        """Calculate total tour length including return."""
        order_node_ = order_node[None, :, :].clone()
        order_flag_ = order_flag[None, :, :].clone()
        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)
        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = 0
        solution = torch.stack((order_node_, order_flag_), dim=3).view(order_node_.size(0), order_node_.size(1), -1)
        batch_size = solution.size(0)
        pomo_size = solution.size(1)
        gathering_index = solution.unsqueeze(3).expand(batch_size, pomo_size, -1, 2)
        seq_expanded = problems[:, None, :, :].expand(batch_size, pomo_size, -1, 2)
        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        travel_distances = segment_lengths.sum(2)
        return travel_distances

    def cal_length_total2(self, problems, order_node, order_flag):
        """Calculate total tour length (batch version)."""
        order_node_ = order_node.clone()
        order_flag_ = order_flag.clone()
        index_small = torch.le(order_flag_, 0.5)
        index_bigger = torch.gt(order_flag_, 0.5)
        order_flag_[index_small] = order_node_[index_small]
        order_flag_[index_bigger] = 0
        solution = torch.stack((order_node_, order_flag_), dim=3).view(order_node_.size(0), order_node_.size(1), -1)
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
        current_node = self.current_node[:, :, None, None].expand(self.batch_size, self.pomo_size, 1, self.problem_size + 1)
        cur_dist = self.dist[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size + 1, self.problem_size + 1).gather(2, current_node).squeeze(2)
        return cur_dist

    def make_dataset(self, filename, episode):
        """Load CVRP dataset from file."""
        def tow_col_nodeflag(node_flag):
            tow_col_node_flag = []
            V = int(len(node_flag) / 2)
            for i in range(V):
                tow_col_node_flag.append([node_flag[i], node_flag[V + i]])
            return tow_col_node_flag

        raw_data_nodes = []
        raw_data_capacity = []
        raw_data_demand = []
        raw_data_cost = []
        raw_data_node_flag = []

        for line in open(filename, "r").readlines()[0:episode]:
            line = line.split(",")
            depot_index = int(line.index('depot'))
            customer_index = int(line.index('customer'))
            capacity_index = int(line.index('capacity'))
            demand_index = int(line.index('demand'))
            cost_index = int(line.index('cost'))
            node_flag_index = int(line.index('node_flag'))
            depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
            customer = [[float(line[idx]), float(line[idx + 1])] for idx in range(customer_index + 1, capacity_index, 2)]
            loc = depot + customer
            capacity = int(float(line[capacity_index + 1]))
            demand = [0] + [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
            cost = float(line[cost_index + 1])
            node_flag = [int(line[idx]) for idx in range(node_flag_index + 1, len(line))]
            node_flag = tow_col_nodeflag(node_flag)
            raw_data_nodes.append(loc)
            raw_data_capacity.append(capacity)
            raw_data_demand.append(demand)
            raw_data_cost.append(cost)
            raw_data_node_flag.append(node_flag)

        raw_data_nodes = torch.tensor(raw_data_nodes, requires_grad=False)
        raw_data_demand = torch.tensor(raw_data_demand, requires_grad=False) / torch.tensor(raw_data_capacity, requires_grad=False)[:, None]
        raw_data_cost = torch.tensor(raw_data_cost, requires_grad=False)
        raw_data_node_flag = torch.tensor(raw_data_node_flag, requires_grad=False)

        return raw_data_nodes, raw_data_demand, raw_data_cost, raw_data_node_flag
