import math
import numpy as np 
import torch
from torch import nn
import torch.nn.functional as F



"""
平方欧氏距离 x^2 + y^2 - 2xy
"""
def pairwise_distance(x): # x:(B, N, D) 
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    x_inner = -2*torch.matmul(x, x.transpose(2, 1))
    x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
    return x_square + x_inner + x_square.transpose(2, 1)


class dense_knn_matrix(nn.Module):
    def __init__(self, k=16):
        super(dense_knn_matrix, self).__init__()
        self.k = k

    def forward(self, x): # x:(B, N, D) 
        batch_size, n_points, n_dims = x.shape
        dist = pairwise_distance(x)
        _, nn_idx = torch.topk(-dist, k=self.k) # b, n, k
        ######
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, self.k, 1).transpose(2, 1) # (b,N,k)
        return torch.stack((nn_idx, center_idx), dim=0) # (2, b,196,k)        



# import hnswlib

# """
# 仅用于batch_size = 1
# """
# def ANN_matrix(x, k=16, ef_construction= 200, M=32, random_result=True, random_seed = 0): # x:(B, N, D)  
#     batch_size, n_points, n_dims = x.shape
    
#     p = hnswlib.Index(space='l2', dim=n_dims)
#     if random_result:
#         p.init_index(max_elements=n_points, ef_construction= ef_construction, M=M)
#     else:
#         p.init_index(max_elements=n_points, ef_construction= ef_construction, M=M, random_seed= random_seed)
#     p.add_items(x.squeeze(0).cpu().detach().numpy())
#     labels, _ = p.knn_query(x.squeeze(0).cpu().detach().numpy(), k=k) # (N,k)
#     nn_idx = torch.from_numpy(labels.astype(np.int32)).unsqueeze(0).to(x.device) # # (B,N,k)

#     center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1) # (b,N,k)
    
#     return torch.stack((nn_idx, center_idx), dim=0)# (2,B,N,k)




# class DenseDilatedKnnGraph(nn.Module):
#     """
#     Find the neighbors' indices based on dilated knn
#     """
#     def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0, random_result=False):
#         super(DenseDilatedKnnGraph, self).__init__()
#         self.dilation = dilation
#         self.stochastic = stochastic
#         self.epsilon = epsilon
#         self.k = k
#         self._dilated = DenseDilated(k, dilation, stochastic, epsilon)
#         self.random_result = random_result

#     def forward(self, x): # x:(B, N, D)  
#         #normalize
#         x = F.normalize(x, p=2.0, dim=-1)  # x:(B, N, D)        
#         # ANN
#         edge_index = ANN_matrix(x, self.k * self.dilation, random_result=self.random_result)# (2, B, N, k) [nn_idx, center_idx]
#         return self._dilated(edge_index) # (2, B, 196, k/dilation)





"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x) # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A



class atten_matrix(nn.Module):
    def __init__(self, in_channels, gate=True, k=16, dropout = 0.):
        super(atten_matrix, self).__init__()
        if gate:
            self.attention_net = Attn_Net_Gated(L = in_channels, D = in_channels, dropout = dropout, n_classes = 1)
        else:
            self.attention_net = Attn_Net(L = in_channels, D = in_channels, dropout = dropout, n_classes = 1)
        
        self.k = k

    def forward(self, x): # x:(B, N, D) 
        batch_size, n_points, n_dims = x.shape

        A = self.attention_net(x)  # A:(B, N, 1) 
        A = F.softmax(A, dim=1)
        dist_matrix = torch.cdist(A, A, p=2)  # (B, N, N) 
        _, nn_idx = torch.topk(dist_matrix, k=self.k, largest=False) # 找到每个点的 K 近邻 (B,N,k) 

        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, self.k, 1).transpose(2, 1) # (b,N,k)
        
        return torch.stack((nn_idx, center_idx), dim=0)# (2,B,N,k)



class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list

    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
    def forward(self, edge_index): # (2,B,N,k)
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index



class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, in_channels, k=9, dilation=1, stochastic=False, epsilon=0.0, dropout = 0., knn="gateAtten", gate=True):
        super(DenseDilatedKnnGraph, self).__init__()
        self.stochastic = stochastic
        self.epsilon = epsilon
        if knn =="gateAtten":
            self.knn_matrix = atten_matrix(in_channels, gate=gate, k=k * dilation, dropout =dropout)
        elif knn =="l2":
            self.knn_matrix = dense_knn_matrix(k * dilation)
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)
   
    def forward(self, x): # x:(B, N, D)  
        edge_index = self.knn_matrix(x)# (2, B, N, k) [nn_idx, center_idx]
        return self._dilated(edge_index) # (2, B, 196, k/dilation)        