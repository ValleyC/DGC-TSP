"""
TSP Sub-solver Model (ICAM).

Based on UDC's TSPModel using AFT (Attention Free Transformer).
Solves SHPP (open path) sub-problems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TSPModel(nn.Module):
    """TSP sub-problem solver using AFT attention."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None
        self.log_scale = None

    def pre_forward(self, reset_state):
        """Pre-compute encodings."""
        dist = reset_state.dist
        self.encoded_nodes = self.encoder(reset_state.problems, dist, reset_state.log_scale)
        self.decoder.set_kv(self.encoded_nodes)
        self.log_scale = reset_state.log_scale

    def forward(self, state, cur_dist):
        """Select next node."""
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        device = state.BATCH_IDX.device

        if state.current_node is None:
            # First step: half start from 0, half from last
            selected = torch.cat([
                torch.zeros(pomo_size // 2, device=device)[None, :].expand(batch_size, -1),
                (state.ninf_mask.size(-1) - 1) * torch.ones(pomo_size // 2, device=device)[None, :].expand(batch_size, -1)
            ], dim=-1).long()
            prob = torch.ones(size=(batch_size, pomo_size), device=device)

            encoded_first_node = _get_encoding(
                self.encoded_nodes,
                torch.cat([
                    (state.ninf_mask.size(-1) - 1) * torch.ones(pomo_size // 2, device=device)[None, :].expand(batch_size, -1),
                    torch.zeros(pomo_size // 2, device=device)[None, :].expand(batch_size, -1)
                ], dim=-1).long()
            )
            self.decoder.set_q1(encoded_first_node)
        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            probs = self.decoder(encoded_last_node, cur_dist, self.log_scale, ninf_mask=state.ninf_mask)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
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


class TSP_Encoder(nn.Module):
    """Encoder for TSP sub-problems."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        encoder_layer_num = model_params['encoder_layer_num']

        self.embedding = nn.Linear(2, embedding_dim)
        self.embedding_out = nn.Linear(embedding_dim, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data, dist, log_scale):
        embedded_input = self.embedding(data)
        # Mark boundary nodes
        embedded_input[:, -1, :] = self.embedding_out(embedded_input[:, -1, :].clone())
        embedded_input[:, 0, :] = self.embedding_out(embedded_input[:, 0, :].clone())

        out = embedded_input
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
        self.multi_head_combine = nn.Linear(embedding_dim, embedding_dim)

        self.addAndNorm1 = AddAndNorm(**model_params)
        self.feedForward = FeedForward(**model_params)
        self.addAndNorm2 = AddAndNorm(**model_params)

        self.AFT_dist_alpha_1 = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

    def forward(self, input1, negative_dist, log_scale):
        q = self.Wq(input1)
        k = self.Wk(input1)
        v = self.Wv(input1)

        # AFT
        sigmoid_q = torch.sigmoid(q)
        alpha_dist_bias_scale = log_scale * self.AFT_dist_alpha_1 * negative_dist
        bias = torch.exp(alpha_dist_bias_scale) @ torch.mul(torch.exp(k), v)
        weighted = bias / (torch.exp(alpha_dist_bias_scale) @ torch.exp(k))
        out = torch.mul(sigmoid_q, weighted)

        out1 = self.addAndNorm1(input1, out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNorm2(out1, out2)
        return out3


class TSP_Decoder(nn.Module):
    """Decoder for TSP sub-problems."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']

        self.Wq_first = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.k = None
        self.v = None
        self.single_head_key = None
        self.q_first = None

        self.dist_alpha_1 = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.AFT_dist_alpha_1 = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

    def set_kv(self, encoded_nodes):
        self.k = self.Wk(encoded_nodes)
        self.v = self.Wv(encoded_nodes)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def set_q1(self, encoded_q1):
        self.q_first = self.Wq_first(encoded_q1)

    def forward(self, encoded_last_node, cur_dist, log_scale, ninf_mask):
        q_last = self.Wq_last(encoded_last_node)
        q = self.q_first + q_last
        cur_dist_out = -1 * cur_dist

        # AFT
        sigmoid_q = torch.sigmoid(q)
        alpha_dist_bias_scale = log_scale * self.AFT_dist_alpha_1 * cur_dist_out
        alpha_dist_bias_scale = alpha_dist_bias_scale + ninf_mask
        bias = torch.exp(alpha_dist_bias_scale) @ torch.mul(torch.exp(self.k), self.v)
        weighted = bias / (torch.exp(alpha_dist_bias_scale) @ torch.exp(self.k))
        AFT_out = torch.mul(sigmoid_q, weighted)

        # Probabilities
        score = torch.matmul(AFT_out, self.single_head_key)
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        score_scaled = score / sqrt_embedding_dim
        score_scaled = score_scaled + log_scale * self.dist_alpha_1 * cur_dist_out

        logit_clipping = self.model_params['logit_clipping']
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        return probs


class AddAndNorm(nn.Module):
    """Add & Normalize."""

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
