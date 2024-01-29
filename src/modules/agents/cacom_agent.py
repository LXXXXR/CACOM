import random

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .rnn_agent import LsqQuan


class CACOM_Agent(nn.Module):
    def __init__(self, input_shape, args):
        super(CACOM_Agent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.encode_dim = args.encode_dim
        self.request_dim = args.request_dim
        self.response_dim = args.response_dim
        self.obs_segs = args.obs_segs
        # assert self.request_dim + self.response_dim == self.n_actions

        NN_HIDDEN_SIZE = args.nn_hidden_size
        NN_HIDDEN_MULTI = args.nn_hidden_multi
        activation_func = nn.LeakyReLU()

        self.input_encoders = nn.ModuleList()
        obs_seg_dim = 0
        obs_seg_num = 0
        for seg_num, seg_len in self.obs_segs:
            obs_seg_dim = obs_seg_dim + seg_num * seg_len
            obs_seg_num = obs_seg_num + seg_num
            self.input_encoders.append(nn.Linear(seg_len, self.encode_dim))

        assert obs_seg_dim == input_shape
        # self.input_norm_1 = nn.LayerNorm(self.encode_dim)
        self.input_kqv = nn.Linear(self.encode_dim, self.encode_dim * 3)
        # self.input_norm_2 = nn.LayerNorm(self.encode_dim)
        self.input_fc = nn.Sequential(
            nn.Linear(self.encode_dim, NN_HIDDEN_MULTI * self.encode_dim),
            activation_func,
            nn.Linear(NN_HIDDEN_MULTI * self.encode_dim, self.encode_dim),
        )
        self.request_generator = nn.Linear(
            args.rnn_hidden_dim + obs_seg_num * self.encode_dim, args.request_dim
        )

        # self.response_norm_1 = nn.LayerNorm(self.encode_dim)
        self.response_kv = nn.Linear(self.encode_dim, 2 * self.encode_dim)
        self.response_q = nn.Linear(args.request_dim, args.encode_dim)
        # self.response_norm_2 = nn.LayerNorm(self.encode_dim)
        self.response_fc = nn.Sequential(
            nn.Linear(self.encode_dim, NN_HIDDEN_MULTI * self.encode_dim),
            activation_func,
            nn.Linear(NN_HIDDEN_MULTI * self.encode_dim, args.response_dim),
        )

        self.actor_feat_kqv = nn.Sequential(
            activation_func, nn.Linear(self.encode_dim, 3 * self.encode_dim)
        )
        self.actor_msg_kqv = nn.Linear(args.response_dim, 3 * self.encode_dim)
        self.actor_linear = nn.Linear(
            (self.n_agents - 1 + obs_seg_num) * self.encode_dim, args.rnn_hidden_dim
        )
        self.actor_rnn = nn.GRUCell(
            input_size=args.rnn_hidden_dim, hidden_size=args.rnn_hidden_dim
        )
        self.actor_linear_2 = nn.Linear(args.rnn_hidden_dim, self.n_actions)

        self.pred_feat_kv = nn.Linear(self.encode_dim, 2 * self.encode_dim)
        self.pred_msg_kqv = nn.Linear(args.response_dim, 3 * self.encode_dim)
        self.pred_linear = nn.Linear(
            (self.n_agents - 1) * self.encode_dim, (self.n_agents - 1) * self.n_actions
        )

        if hasattr(self.args, "discrete_bits"):
            self.req_quan = LsqQuan(bit=self.args.discrete_bits)
            self.res_quan = LsqQuan(bit=self.args.discrete_bits)

    def init_hidden(self):
        return self.actor_linear.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(
        self,
        inputs,
        hidden_state,
        bs,
        exp_gate,
        all_through=False,
        test_mode=False,
        **kwargs
    ):
        """_summary_

        Args:
            inputs (_type_): (bs * n_agents, input_shape)
            hidden_state (_type_): (bs, n_agents, rnn_hidden_dim)
            bs (_type_): int
            test_mode (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        # (bs, n_agents, nun_seg, encode_dim),
        # (bs, reply_agents, request_agents, request_dim)
        feat, requests = self.pre_comm(inputs, hidden_state, bs)

        # (bs, reply_agents, request_agents, response_dim)
        response = self.comm_response(feat, requests)
        _, response_mask = exp_gate(
            requests, feat, test_mode=True, all_through=all_through
        )
        response = response * response_mask

        # (bs, request_agents, reply_agents, response_dim)
        response = response.permute((0, 2, 1, 3))
        mask = th.eye(self.n_agents, dtype=bool, device=self.args.device)[
            None, :, :, None
        ].repeat(bs, 1, 1, self.response_dim)
        # (bs, n_agents, n_agents - 1, response_dim)
        response = response.masked_select(~mask).reshape(
            (bs, self.n_agents, -1, self.response_dim)
        )

        if hasattr(self.args, "discrete_bits"):
            response = self.res_quan(response)

        h, return_q = self.after_comm(feat, response, hidden_state, bs)
        returns = {}
        if "train_mode" in kwargs and kwargs["train_mode"]:
            if hasattr(self.args, "pred_weight") and self.args.pred_weight > 0:
                returns["aux_loss"] = self.args.pred_weight * self.calculate_aux_loss(
                    bs, feat, response, return_q
                )

        return return_q, h, returns, response_mask.float().mean().detach()

    def pre_comm(self, inputs, hidden_state, bs):
        # TODO: try to avoid loop later
        obs_seg_dim = 0
        obs_encoded = []
        for i, (seg_num, seg_len) in enumerate(self.obs_segs):
            obs_seg = inputs[:, obs_seg_dim : obs_seg_dim + seg_num * seg_len].clone()
            obs_seg = obs_seg.reshape((bs, self.n_agents, seg_num, seg_len))
            obs_seg = self.input_encoders[i](obs_seg)

            obs_encoded.append(obs_seg)
            obs_seg_dim = obs_seg_dim + seg_num * seg_len

        obs_encoded = th.cat(obs_encoded, dim=2)
        obs_encoded = obs_encoded.reshape((bs, self.n_agents, -1, self.encode_dim))
        # kqv = self.input_kqv(self.input_norm_1(obs_encoded))
        kqv = self.input_kqv(obs_encoded)
        k, q, v = th.chunk(kqv, 3, dim=-1)
        feat_scores = th.matmul(q, k.permute((0, 1, 3, 2))) / np.sqrt(self.encode_dim)
        feat_weights = F.softmax(feat_scores, dim=-1)[:, :, :, :, None]
        # (bs, n_agents, num_seg, encode_dim)
        feat = (v[:, :, None, :, :] * feat_weights).sum(dim=-2)
        feat = obs_encoded + feat
        # feat = feat + self.input_fc(self.input_norm_2(feat))
        feat = feat + self.input_fc(feat)

        requests = self.request_generator(
            th.cat([hidden_state, feat.reshape((bs, self.n_agents, -1))], dim=-1)
        )
        if hasattr(self.args, "discrete_bits"):
            requests = self.req_quan(requests)
        # (bs, reply_agents, request_agents, request_dim)
        requests = requests[:, None, :, :].repeat(1, self.n_agents, 1, 1)
        # mask = th.eye(self.n_agents, dtype=bool)[None, :,:, None].repeat(bs, 1, 1, self.request_dim)
        # (bs, n_agents, n_agents - 1, request_dim)
        # requests = requests.masked_select(~mask)

        return feat, requests

    def comm_response(self, feat, requests):
        # kv = self.response_kv(self.response_norm_1(feat))
        kv = self.response_kv(feat)
        # (bs, reply_agents, num_seg, encode_dim/response_dim)
        # k, v = th.split(kv, [self.encode_dim, self.response_dim], dim=-1)
        k, v = th.chunk(kv, 2, dim=-1)
        # (bs, reply_agents, encode_dim, request_agents)
        q = self.response_q(requests).permute((0, 1, 3, 2))
        # (bs, n_agents, num_seg, request_agents)
        response_scores = th.matmul(k, q) / np.sqrt(self.encode_dim)
        response_weights = F.softmax(response_scores, dim=-2)
        # (bs, reply_agents, request_agents, response_dim)
        response = (v[:, :, :, None, :] * response_weights[:, :, :, :, None]).sum(
            dim=-3
        )
        # response = self.response_fc(self.response_norm_2(response))
        response = self.response_fc(response)

        return response

    def after_comm(self, feat, response, hidden_state, bs):
        msg_kqv = self.actor_msg_kqv(response)
        msg_k, msg_q, msg_v = th.chunk(msg_kqv, 3, dim=-1)
        kqv = self.actor_feat_kqv(feat)
        k, q, v = th.chunk(kqv, 3, dim=-1)
        k = th.cat([k, msg_k], dim=2).permute((0, 1, 3, 2))
        q = th.cat([q, msg_q], dim=2)
        # (bs, n_agents, 1， num_seg + n_agents - 1, encode_dim)
        v = th.cat([v, msg_v], dim=2)[:, :, None, :, :]
        # (bs, n_agents, num_seg + n_agents - 1, num_seg + n_agents - 1)
        scores = th.matmul(q, k) / np.sqrt(self.encode_dim)
        soft_weights = F.softmax(scores, dim=-1)[:, :, :, :, None]
        # (bs, n_agents, num_seg + n_agents - 1, encode_dim)
        x = (
            (v * soft_weights)
            .sum(dim=-2)
            .reshape((bs, self.n_agents, -1, self.encode_dim))
        )
        # residual connection for feature
        x[:, :, : feat.shape[2], :] = feat + x[:, :, : feat.shape[2], :]

        x = self.actor_linear(x.reshape((bs * self.n_agents, -1)))
        hidden_state = hidden_state.reshape((bs * self.n_agents, -1))
        h = self.actor_rnn(x, hidden_state)
        h = h.reshape((bs, self.n_agents, -1))

        return_q = self.actor_linear_2(h).reshape((-1, self.n_actions))

        return h, return_q

    def cal_gate_labels(self, inputs, hidden_state, bs, exp_gate):
        with th.no_grad():
            feat, requests = self.pre_comm(inputs, hidden_state, bs)
            # (bs, reply_agents, request_agents, response_dim)
            response_ori = self.comm_response(feat, requests)

        # (bs, reply_agents, request_agents, response_dim)
        response_probs, response_mask = exp_gate(requests, feat, test_mode=False)
        response_mask = response_mask.detach()

        idx = random.randrange(self.n_agents)
        probs = response_probs[:, idx, [i for i in range(self.n_agents) if i != idx], :]

        with th.no_grad():
            response = response_ori * response_mask
            # (bs, request_agents, reply_agents, response_dim)
            response = response.permute((0, 2, 1, 3))
            mask = th.eye(self.n_agents, dtype=bool, device=self.args.device)[
                None, :, :, None
            ].repeat(bs, 1, 1, self.response_dim)
            # (bs, n_agents, n_agents - 1, response_dim)
            response = response.masked_select(~mask).reshape(
                (bs, self.n_agents, -1, self.response_dim)
            )
            if hasattr(self.args, "discrete_bits"):
                response = self.res_quan(response)
            h, _ = self.after_comm(feat, response, hidden_state, bs)

            response_mask[:, idx, :, :] = 1
            response = response_ori * response_mask
            # (bs, request_agents, reply_agents, response_dim)
            response = response.permute((0, 2, 1, 3))
            mask = th.eye(self.n_agents, dtype=bool, device=self.args.device)[
                None, :, :, None
            ].repeat(bs, 1, 1, self.response_dim)
            # (bs, n_agents, n_agents - 1, response_dim)
            response = response.masked_select(~mask).reshape(
                (bs, self.n_agents, -1, self.response_dim)
            )
            if hasattr(self.args, "discrete_bits"):
                response = self.res_quan(response)
            _, q_pos = self.after_comm(feat, response, hidden_state, bs)
            q_pos = q_pos.reshape((bs, self.n_agents, -1))

            response_mask[:, idx, :, :] = 0
            response = response_ori * response_mask
            # (bs, request_agents, reply_agents, response_dim)
            response = response.permute((0, 2, 1, 3))
            mask = th.eye(self.n_agents, dtype=bool, device=self.args.device)[
                None, :, :, None
            ].repeat(bs, 1, 1, self.response_dim)
            # (bs, n_agents, n_agents - 1, response_dim)
            response = response.masked_select(~mask).reshape(
                (bs, self.n_agents, -1, self.response_dim)
            )
            if hasattr(self.args, "discrete_bits"):
                response = self.res_quan(response)
            _, q_neg = self.after_comm(feat, response, hidden_state, bs)
            q_neg = q_neg.reshape((bs, self.n_agents, -1))

        # (bs, n_agents, 2/n_actions)
        return h, probs, q_pos, q_neg, idx

    def calculate_aux_loss(self, bs, feat, response, return_q):
        """_summary_

        Args:
            h (_type_): _description_
            bs (_type_): _description_
            feat (_type_): _description_
            response (_type_): _description_
            q (_type_): _description_

        Returns:
            _type_: _description_
        """

        msg_kqv = self.pred_msg_kqv(response)
        # (bs, n_agents, n_agents - 1, encode_dim)
        msg_k, q, msg_v = th.chunk(msg_kqv, 3, dim=-1)
        kv = self.pred_feat_kv(feat)
        k, v = th.chunk(kv, 2, dim=-1)
        k = th.cat([k, msg_k], dim=2).permute((0, 1, 3, 2))
        # (bs, n_agents, 1, num_seg + n_agents - 1, encode_dim)
        v = th.cat([v, msg_v], dim=2)[:, :, None, :, :]
        # (bs, n_agents, n_agents - 1, num_seg + n_agents - 1)
        scores = th.matmul(q, k) / np.sqrt(self.encode_dim)
        soft_weights = F.softmax(scores, dim=-1)[:, :, :, :, None]
        # (bs, n_agents, n_agents - 1, encode_dim)
        x = (v * soft_weights).sum(dim=-2).reshape((bs * self.n_agents, -1))
        # (bs, n_agents, (n_agents - 1) * n_actions)
        x = self.pred_linear(x).reshape((bs, self.n_agents, -1, self.n_actions))
        return_q = return_q.reshape((bs, self.n_agents, self.n_actions))[
            :, None, :, :
        ].repeat(1, self.n_agents, 1, 1)
        mask = th.eye(self.n_agents, dtype=bool, device=self.args.device)[
            None, :, :, None
        ].repeat(bs, 1, 1, self.n_actions)
        # (bs, n_agents, n_agents - 1, encode_dim)
        return_q = return_q.masked_select(~mask).reshape(
            (bs, self.n_agents, -1, self.n_actions)
        )
        loss = F.mse_loss(x, return_q.detach())

        return loss


class ExpGate(nn.Module):
    def __init__(self, args):
        super(ExpGate, self).__init__()
        self.encode_dim = args.encode_dim
        self.obs_segs = args.obs_segs
        NN_HIDDEN_MULTI = args.nn_hidden_multi
        activation_func = nn.LeakyReLU()
        seq_len = 0
        for seg_num, _ in self.obs_segs:
            seq_len = seq_len + seg_num

        self.k = nn.Linear(self.encode_dim, self.encode_dim)
        self.q = nn.Linear(args.request_dim, self.encode_dim)
        self.gate = nn.Linear(seq_len, 2)

    def forward(self, requests, feat, test_mode, all_through=False):
        """_summary_

        Args:
            requests (_type_): (bs, reply_agents, request_agents, request_dim)
            feat (_type_): (bs, n_agents, num_seg, encode_dim)

        Returns:
            _type_: _description_
        """
        # masks = None

        if all_through:
            probs = None
            masks = th.ones_like(requests[:, :, :, [0]])

        else:
            if test_mode:
                with th.no_grad():
                    # (bs, n_agents, num_seg, encode_dim)
                    k = self.k(feat).permute((0, 1, 3, 2))
                    # (bs, reply_agents, request_agents, encode_dim)
                    q = self.q(requests)
                    # (bs, reply_agents, request_agents, num_seg)
                    weights = th.matmul(q, k)
                    # (bs, reply_agents, request_agents, 2)
                    probs = self.gate(weights)
                    probs = F.softmax(probs, dim=-1)[:, :, :, [0]]
                    # (bs, reply_agents, request_agents, 1)
                    masks = probs > 0.5
            else:
                k = self.k(feat).permute((0, 1, 3, 2))
                # (bs, reply_agents, request_agents, encode_dim)
                q = self.q(requests)
                # (bs, reply_agents, request_agents, num_seg)
                weights = th.matmul(q, k)
                # (bs, reply_agents, request_agents, 2)
                probs = self.gate(weights)
                # (bs, reply_agents, request_agents, 1)
                masks = F.softmax(probs, dim=-1)[:, :, :, [0]] > 0.5

        return probs, masks
