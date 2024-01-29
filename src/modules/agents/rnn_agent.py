import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


def grad_scale(s, scale):
    y = s
    y_grad = s * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

class LsqQuan(nn.Module):
    """https://github.com/zhutmost/lsq-net/blob/master

    Args:
        Quantizer (_type_): _description_
    """

    def __init__(self, bit=8, symmetric=True, per_channel=False):
        super().__init__()
        if symmetric:
            # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
            self.thd_neg = -(2 ** (bit - 1)) + 1
            self.thd_pos = 2 ** (bit - 1) - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.thd_neg = -(2 ** (bit - 1))
            self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = nn.Parameter(th.ones(1))

    # def init_from(self, x, *args, **kwargs):
    #     if self.per_channel:
    #         self.s = nn.Parameter(
    #             x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True)
    #             * 2
    #             / (self.thd_pos**0.5)
    #         )
    #     else:
    #         self.s = nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos**0.5))

    def forward(self, x):
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        x = x / s_scale
        x = th.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x

