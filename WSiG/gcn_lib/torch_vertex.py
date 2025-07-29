import numpy as np
import math
import torch
from torch import nn
from .torch_nn import norm_layer, MLP, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
# from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath


class MRConv1d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv1d, self).__init__()
        self.nn = MLP([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None): # x:(B, N, D) edge_index:(2, B, N, k)
        x = x.transpose(2, 1) # x:(B, D, N)
        x_i = batched_index_select(x, edge_index[1]) # center_idx: (B, N, k) x_i: (B, D, N, k)
        x_j = batched_index_select(x, edge_index[0]) # knn_idx: (B, N, k) x_j: (B, D, N, k)
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=False) # (B, D, N)
        b, c, n = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n)# (B, 2D, N)
        x = x.transpose(2, 1) # (B, N, 2D)
        return self.nn(x) # (B, N, out_channels)


class EdgeConv1d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv1d, self).__init__()
        self.nn = MLP([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):# x:(B, N, D)
        x = x.transpose(2, 1) # x:(B, D, N)
        x_i = batched_index_select(x, edge_index[1]) # x_i: (B, D, N, k)
        x_j = batched_index_select(x, edge_index[0]) # x_j: (B, D, N, k)
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1).permute(0, 2, 3, 1)), -2, keepdim=False) # (B, N, D)
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = MLP([in_channels, in_channels], act, norm, bias)
        self.nn2 = MLP([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):# x:(B, N, D)
        x = x.transpose(2, 1) # x:(B, D, N)
        x_j = batched_index_select(x, edge_index[0]) # x_j: (B, D, N, k)
        x_j, _ = torch.max(self.nn1(x_j.permute(0, 2, 3, 1)), -2, keepdim=False) # (B, N, D)
        return self.nn2(torch.cat([x.transpose(2, 1), x_j], dim=-1))


class GINConv1d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv1d, self).__init__()
        self.nn = MLP([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):# x:(B, N, D)
        x = x.transpose(2, 1) # x:(B, D, N)
        x_j = batched_index_select(x, edge_index[0]) # x_j: (B, D, N, k)
        x_j = torch.sum(x_j, -1, keepdim=False) # x_j: (B, D, N)
        return self.nn((1 + self.eps) * x.transpose(2, 1) + x_j.transpose(2, 1)) # (B, N, D)


class GATConv(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GATConv, self).__init__()
        self.nn = MLP([in_channels*2, 1], act, norm, bias)
        self.act = act_layer('leakyrelu')
        self.nn2 = MLP([in_channels, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):# x:(B, N, D)
        x = x.transpose(2, 1) # x:(B, D, N)
        x_i = batched_index_select(x, edge_index[1]) # x_i: (B, D, N, k)
        x_j = batched_index_select(x, edge_index[0]) # x_j: (B, D, N, k)
        scores = self.act(self.nn(torch.cat([x_i, x_j], dim=1).permute(0, 2, 3, 1)))  # x_j: (B, N, k, 1)
        p_attn = F.softmax(scores, dim=-2) # (B, N, k, 1)      
        return self.nn2(torch.sum(x_j.permute(0, 2, 3, 1)* p_attn, -2, keepdim=False)) # (B, N, D)


class AttenConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(AttenConv1d, self).__init__()
        self.nn = MLP([in_channels, out_channels], act, norm, bias)

    def forward(self, x, edge_index):# x:(B, N, D)
        x = x.transpose(2, 1) # x:(B, D, N)
        x_i = batched_index_select(x, edge_index[1]) # center_idx: (B, N, k) x_i: (B, D, N, k)
        x_j = batched_index_select(x, edge_index[0]) # knn_idx: (B, N, k) x_j: (B, D, N, k)
        x, x_i, x_j = x.transpose(2, 1), x_i.permute(0, 2, 3, 1), x_j.permute(0, 2, 3, 1) # x:(B, N, D) x_i: (B, N, k, D) x_j: (B, N, k, D) 
        
        scores = torch.matmul(x_i, x_j.transpose(-2, -1)) / math.sqrt(x_i.size(-1)) # (B, N, k, k) 
        p_attn = F.softmax(scores, dim=-1) # (B, N, k, k) 
        h_k = torch.sum(torch.matmul(p_attn, x_j), -2, keepdim=False) # (B, N, k, D) -> (B, N, D)
        return self.nn(x+h_k)



class DyGraphConv1d(nn.Module):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, dropout = 0., knn="gateAtten", gate=True):
        super(DyGraphConv1d, self).__init__()
        self.k = kernel_size
        self.d = dilation
        self.dilated_knn_graph = DenseDilatedKnnGraph(in_channels, kernel_size, dilation, stochastic, epsilon, dropout, knn, gate)

        if conv == 'edge':
            self.gconv = EdgeConv1d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv1d(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv1d(in_channels, out_channels, act, norm, bias)
        elif conv == 'gat':
            self.gconv = GATConv(in_channels, out_channels, act, norm, bias)
        elif conv == 'attention':
            self.gconv = AttenConv1d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x):
        edge_index = self.dilated_knn_graph(x) # (2, B, N, k)
        x = self.gconv(x, edge_index) # (B, N, out_channels)
        return x.contiguous(), edge_index



class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, drop_path=0.0, dropout=0., knn="gateAtten", gate=True):    
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.fc1 = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            norm_layer(norm, in_channels),
        )        
        self.graph_conv = DyGraphConv1d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                        act, norm, bias, stochastic, epsilon, dropout, knn, gate)
        self.fc2 = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            norm_layer(norm, in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x): # x:(B, N, D)
        _tmp = x
        x = self.fc1(x) # x:(B, N, D)
        x, edge_index = self.graph_conv(x) # (B, N, 2D)
        x = self.fc2(x) # x:(B, N, D)
        x = self.drop_path(x) + _tmp # x:(B, out_channels, D)
        return x, edge_index