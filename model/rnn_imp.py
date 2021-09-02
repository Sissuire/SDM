import torch
import torch.nn as nn
import math


class RNN_trial(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bias=True, dropout=0.0, batch_first=True):
        super(RNN_trial, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bias = bias
        self.batch_first = batch_first

        #self.cell_1 = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias).cuda()
        ####self.cell_2 = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size, bias=bias).cuda()
        self.cell_3 = nn.LSTMCell(input_size=hidden_size, hidden_size=1, bias=bias).cuda()

        #self.cell_1 = LSTM_cell(input_size=input_size, hidden_size=hidden_size, bias=bias).cuda()
        #self.cell_3 = LSTM_cell(input_size=hidden_size, hidden_size=1, bias=bias).cuda()

        self.cell_1 = LSTM_cell_mem(input_size=input_size, hidden_size=hidden_size, bias=bias).cuda()
        #self.cell_3 = LSTM_cell_mem(input_size=hidden_size, hidden_size=1, bias=bias).cuda()

        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, hidden_state):
        """
        :param x: [batch_size, seq_len, input_size] or [seq_len, batch_size, input_size]
        :param hidden_state: (h, c) | [num_layer, batch_size, hidden*directions]
        :return:  out shape [batch_size, seq_len, hidden_size]
        """
        h, c, _ = hidden_state

        if self.batch_first:
            x = x.permute((1, 0, 2))

        seq_len = x.shape[0]
        batch_size = x.shape[1]
        out_1, out_2, ret = [], [], []

        worst_score = -100 * torch.ones(batch_size, 1).cuda()

        x_in = x.clone()

        # 1st layer
        h0, c0 = h[0], c[0]
        m = h0.clone()
        for j in range(seq_len):
            h0, c0 = self.cell_1(x_in[j], (h0, c0, m))
            out_1.append(h0.clone())

            cond = x_in[j] > worst_score
            m = torch.where(cond, h0, m)
            worst_score = torch.where(cond, x_in[j], worst_score)
 
        # 2nd layer
        #h0, c0 = h[1], c[1]
        #for j in range(seq_len):
        #    h0, c0 = self.cell_2(out_1[j], (h0, c0))
        #    out_2.append(h0.clone())

        h0, c0 = h[2], c[2]
        for j in range(seq_len):
            h0, c0 = self.cell_3(out_1[j], (h0, c0))
            ret.append(h0.clone())

        # =====================================
        # =========== self-attention ==========
        # =====================================
        # weight = []
        denom = math.sqrt(self.hidden_size)
        weight = torch.einsum('bs,lbs->lb', [m, torch.stack(out_1)]) / denom
        weight = self.softmax(weight)

        ret = torch.sum(torch.stack(ret).view(weight.shape) * weight, dim=0, keepdim=True).permute((1, 0))

        return ret


class LSTM_cell(nn.Module):
    """
    Implementation of basic LSTM cell

    Args: [input_size, hidden_size]
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0):
        super(LSTM_cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.Drop = nn.Dropout(dropout)

        self.a = nn.Tanh()

        self.i2h = nn.Linear(input_size, 4*hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4*hidden_size, bias=bias)

        self.init_parameters()

    def init_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            nn.init.uniform_(w, -std, std)

    def forward(self, x, hidden_state):
        """"
        x: [batch_size, input_size]
        hidden: (h, c)  |  [batch_size, hidden*directions]
        """

        h, c = hidden_state
        dims = len(h.shape)
        if dims == 3:
            batch_size = h.shape[1]
            h = h.view((batch_size, -1))
            c = c.view((batch_size, -1))
        else:
            batch_size = h.shape[0]

        #
        pre_act = self.i2h(x) + self.h2h(h)

        # i_t = pre_act[:, :self.hidden_size].sigmoid()
        # f_t = pre_act[:, self.hidden_size:2*self.hidden_size].sigmoid()
        # g_t = pre_act[:, 2*self.hidden_size:3*self.hidden_size].tanh()
        # o_t = pre_act[:, 3*self.hidden_size:].sigmoid()

        gates = pre_act[:, :3*self.hidden_size].sigmoid()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2*self.hidden_size]
        o_t = gates[:, -self.hidden_size:]
        # g_t = pre_act[:, -self.hidden_size:].tanh()
        g_t = pre_act[:, -self.hidden_size:]
        g_t = self.a(g_t)

        #if self.training and self.dropout > 0.0:
        #    i_t = self.Drop(i_t)
        #    f_t = self.Drop(f_t)
        #    g_t = self.Drop(g_t)
        #    o_t = self.Drop(o_t)

        c_t = f_t * c + i_t * g_t
        # h_t = c_t.tanh() * o_t  ############ without dropout
        h_t = self.a(c_t) * o_t

        # [num_layer, batch_size, hidden*directions]
        # h_t = h_t.view((1, batch_size, -1))
        # c_t = c_t.view((1, batch_size, -1))
        return h_t, c_t


class LSTM_cell_mem(nn.Module):
    """
    Implementation of assessment-LSTM cell

    Args: [input_size, hidden_size]
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0):
        super(LSTM_cell_mem, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.Drop = nn.Dropout(dropout)

        self.a = nn.Tanh()

        self.i2h = nn.Linear(input_size, 4*hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4*hidden_size, bias=bias)
        self.m2h = nn.Linear(hidden_size, 2*hidden_size, bias=bias)

        self.init_parameters()

    def init_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            nn.init.uniform_(w, -std, std)

    def forward(self, x, hidden_state):
        """"
        :x: [batch_size, input_size]
        hidden: (h, c, m)  |  [batch_size, hidden*directions]
        """

        h, c, m_ = hidden_state
        m = m_.clone()
        dims = len(h.shape)
        if dims == 3:
            batch_size = h.shape[1]
            h = h.view((batch_size, -1))
            c = c.view((batch_size, -1))
        else:
            batch_size = h.shape[0]

        #
        pre_act = self.i2h(x) + self.h2h(h)
        mem_t = self.m2h(m)

        m_w = mem_t[:, :self.hidden_size]
        m_g = mem_t[:, self.hidden_size:].sigmoid()

        # i_t = pre_act[:, :self.hidden_size].sigmoid()
        # f_t = pre_act[:, self.hidden_size:2*self.hidden_size].sigmoid()
        # g_t = pre_act[:, 2*self.hidden_size:3*self.hidden_size].tanh()
        # o_t = pre_act[:, 3*self.hidden_size:].sigmoid()

        gates = pre_act[:, :2*self.hidden_size].sigmoid()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2*self.hidden_size]
        
        ################### o_t = sigmoid(o_t' + m_t)
        #o_t = gates[:, -self.hidden_size:]
        o_t = (pre_act[:, 2*self.hidden_size:3*self.hidden_size] + m_w).sigmoid()

        # g_t = pre_act[:, -self.hidden_size:].tanh()
        g_t = pre_act[:, -self.hidden_size:]
        g_t = self.a(g_t)

        #if self.training and self.dropout > 0.0:
        #    i_t = self.Drop(i_t)
        #    f_t = self.Drop(f_t)
        #    g_t = self.Drop(g_t)
        #    o_t = self.Drop(o_t)

        ##################### c_t = c_t' + m_t
        #c_t = f_t * c + i_t * g_t
        c_t = f_t * c + i_t * g_t + m * m_g

        # h_t = c_t.tanh() * o_t  ############ without dropout
        h_t = self.a(c_t) * o_t

        # [num_layer, batch_size, hidden*directions]
        # h_t = h_t.view((1, batch_size, -1))
        # c_t = c_t.view((1, batch_size, -1))
        return h_t, c_t




