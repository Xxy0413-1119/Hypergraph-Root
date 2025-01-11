from torch.nn.parameter import Parameter
import torch.nn as nn
import torch
import math

# GCN parameters
HGCN_FEATURE_DIM = 1064
HGCN_HIDDEN_DIM = 256
HGCN_OUTPUT_DIM = 64

# Attention parameters
DENSE_DIM = 64
ATTENTION_HEADS = 64

# Final linear layer parameters
FINAL_LINEAR_DIM = 1024

# Training parameters
LEARNING_RATE = 1E-4
WEIGHT_DECAY = 1E-4


class HGraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(HGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = input @ self.weight
        output = adj @ support
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class HGCN(nn.Module):

    def __init__(self):
        super(HGCN, self).__init__()
        self.hgc1 = HGraphConvolution(HGCN_FEATURE_DIM, HGCN_HIDDEN_DIM)
        self.ln1 = nn.LayerNorm(HGCN_HIDDEN_DIM)
        self.hgc2 = HGraphConvolution(HGCN_HIDDEN_DIM, HGCN_OUTPUT_DIM)
        self.ln2 = nn.LayerNorm(HGCN_OUTPUT_DIM)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, adj):
        x = self.hgc1(x, adj)
        x = self.relu1(self.ln1(x))
        x = self.hgc2(x, adj)
        output = self.relu2(self.ln2(x))
        return output


class Attention(nn.Module):

    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = torch.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, input):
        x = torch.tanh(self.fc1(input))
        x = self.fc2(x)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)
        return attention


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.hgcn = HGCN()
        self.attention = Attention(HGCN_OUTPUT_DIM, DENSE_DIM, ATTENTION_HEADS)
        self.fc_1 = nn.Linear(HGCN_OUTPUT_DIM * ATTENTION_HEADS, FINAL_LINEAR_DIM)
        self.fc_2 = nn.Linear(FINAL_LINEAR_DIM, 1)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def forward(self, x, adj):
        x = x.float()  # x.shape = (seq_len, FEATURE_DIM); adj.shape = (seq_len, seq_len)
        adj = adj.float()
        x = self.hgcn(x, adj)
        x = x.unsqueeze(0).float()
        att = self.attention(x)
        node_feature_embedding = att @ x
        node_feature_embedding_con = torch.flatten(node_feature_embedding, start_dim=1)
        fc1_feature = self.fc_1(node_feature_embedding_con)
        output = torch.sigmoid(self.fc_2(fc1_feature)).squeeze(0)
        return output