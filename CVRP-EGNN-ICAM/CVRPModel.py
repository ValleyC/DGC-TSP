"""
CVRP Sub-solver Model (ICAM - Iterative Context Attention Model).

Based on UDC's CVRPModel using AFT (Attention Free Transformer).
This is the "conquering" component that solves sub-problems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CVRPModel(nn.Module):
    """CVRP sub-problem solver using AFT attention."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)
        self.encoded_nodes = None
        self.log_scale = None

    def pre_forward(self, reset_state):
        """Pre-compute encodings."""
        depot_xy = reset_state.depot_xy
        node_xy = reset_state.node_xy
        node_demand = reset_state.node_demand
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        dist = reset_state.dist

        self.log_scale = reset_state.log_scale
        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand, dist, self.log_scale, reset_state.flag_return)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state, cur_dist, flag):
        """Select next node."""
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        device = state.BATCH_IDX.device

        if state.selected_count == 0:
            selected = torch.ones(pomo_size, device=device)[None, :].expand(batch_size, pomo_size).long()
            prob = torch.ones(size=(batch_size, pomo_size), device=device)
            encoded_first_node = _get_encoding(
                self.encoded_nodes,
                (state.ninf_mask.size(-1) - torch.ones(pomo_size, device=device)[None, :].expand(batch_size, pomo_size).long()) * (1 - flag)[:, None].long()
            )
            self.decoder.set_q1(encoded_first_node)
            encoded_depot = _get_encoding(self.encoded_nodes, torch.zeros(pomo_size, device=device)[None, :].expand(batch_size, pomo_size).long())
            self.decoder.set_q2(encoded_depot)
        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            probs = self.decoder(encoded_last_node, state.load, state.left, cur_dist, self.log_scale, ninf_mask=state.ninf_mask)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:
                    with torch.no_grad():
                        selected = (probs.reshape(batch_size * pomo_size, -1) + 1e-10).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    if (prob != 0).all():
                        break
            else:
                selected = probs.argmax(dim=2)
                prob = None

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    """Extract encodings for selected nodes."""
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)
    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    return picked_nodes


class CVRP_Encoder(nn.Module):
    """Encoder for CVRP sub-problems."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        encoder_layer_num = model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(3, embedding_dim)
        self.embedding_out = nn.Linear(embedding_dim, embedding_dim)
        self.embedding_in = nn.Linear(embedding_dim, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand, dist, log_scale, flag):
        embedded_depot = self.embedding_depot(depot_xy)
        embedded_node = self.embedding_node(node_xy_demand)
        id = ((1 - flag) * embedded_node.size(1))[:, None, None].expand(-1, -1, embedded_node.size(-1)).long()
        embedded_node[:, 0, :] = self.embedding_in(embedded_node[:, 0, :].clone())

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        last = self.embedding_out(out.gather(1, id).clone())
        out = out.scatter(1, id, last)
        negative_dist = -1 * dist

        for layer in self.layers:
            out = layer(out, negative_dist, log_scale)

        return out


class EncoderLayer(nn.Module):
    """AFT-based encoder layer."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']

        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

        self.AFT_dist_alpha = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

    def forward(self, input1, negative_dist, log_scale):
        q = self.Wq(input1)
        k = self.Wk(input1)
        v = self.Wv(input1)

        # AFT
        sigmoid_q = torch.sigmoid(q)
        alpha_dist_bias_scale = log_scale * self.AFT_dist_alpha * negative_dist
        bias = torch.exp(alpha_dist_bias_scale) @ torch.mul(torch.exp(k), v)
        weighted = torch.nan_to_num(bias) / torch.nan_to_num(torch.exp(alpha_dist_bias_scale) @ torch.exp(k))
        AFT_out = torch.mul(sigmoid_q, weighted)

        out1 = self.add_n_normalization_1(input1, AFT_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3


class CVRP_Decoder(nn.Module):
    """Decoder for CVRP sub-problems."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']

        self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim + 2, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.dist_alpha_1 = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

        self.k = None
        self.v = None
        self.single_head_key = None
        self.q1 = None
        self.q2 = None

        self.probs_dist_alpha = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.AFT_dist_alpha = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

    def set_kv(self, encoded_nodes):
        self.k = self.Wk(encoded_nodes)
        self.v = self.Wv(encoded_nodes)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def set_q1(self, encoded_q1):
        self.q1 = self.Wq_1(encoded_q1)

    def set_q2(self, encoded_q2):
        self.q2 = self.Wq_2(encoded_q2)

    def forward(self, encoded_last_node, load, left, cur_dist, log_scale, ninf_mask):
        input_cat = torch.cat((encoded_last_node, load[:, :, None], left[:, :, None]), dim=2)
        q_last = self.Wq_last(input_cat)
        q = self.q1 + self.q2 + q_last
        cur_dist_out = -1 * cur_dist

        # AFT
        sigmoid_q = torch.sigmoid(q)
        alpha_dist_bias_scale = log_scale * self.AFT_dist_alpha * cur_dist_out
        alpha_dist_bias_scale = alpha_dist_bias_scale + ninf_mask
        bias = torch.exp(alpha_dist_bias_scale) @ torch.mul(torch.exp(self.k), self.v)
        weighted = torch.nan_to_num(bias) / (torch.nan_to_num(torch.exp(alpha_dist_bias_scale) @ torch.exp(self.k)) + 1e-20)
        AFT_out = torch.mul(sigmoid_q, weighted)

        # Probabilities
        score = torch.matmul(AFT_out, self.single_head_key)
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        score_scaled = score_scaled + log_scale * self.probs_dist_alpha * cur_dist_out
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        return probs


class AddAndInstanceNormalization(nn.Module):
    """Add & Instance Norm."""

    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        added = input1 + input2
        transposed = added.transpose(1, 2)
        normalized = self.norm(transposed)
        return normalized.transpose(1, 2)


class FeedForward(nn.Module):
    """Feed-forward layer."""

    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))
