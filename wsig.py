import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention
from lib.models.WSiG.gcn_lib import Grapher, act_layer, norm_layer
# from gcn_lib import Grapher, act_layer, norm_layer

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

class FFN(nn.Module):
    "Implements FFN equation."
    def __init__(self, in_features, hidden_features=None, out_features=None, norm="layer", act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            norm_layer(norm, hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_features, out_features),
            norm_layer(norm, out_features),
        )        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()        
        

    def forward(self, x):
        x, edge_index =x #
        shortcut= x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x, edge_index    


class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        dim_in = opt.dim_in
        channels = opt.n_filters
        n_classes = opt.n_classes
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        self.n_blocks = opt.n_blocks
        drop_path = opt.drop_path
        dropout = opt.dropout
        pool = opt.pool
        knn = opt.knn
        gate= opt.gate

         
        self.stem = nn.Sequential(nn.Linear(dim_in, channels), act_layer(act))

        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        print('dpr', dpr)
        # num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]  # number of knn's k
        num_knn = [k for _ in range(self.n_blocks)]
        print('num_knn', num_knn)
        max_dilation = 196 // max(num_knn)

        # self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))   

        if opt.use_dilation:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                                bias, stochastic, epsilon, drop_path=dpr[i], dropout= dropout, knn=knn, gate=gate),
                                      FFN(channels, channels * 4, norm=norm, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)]) # 改
        else:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], 1, conv, act, norm,
                                                bias, stochastic, epsilon, drop_path=dpr[i], dropout= dropout, knn=knn, gate=gate),
                                      FFN(channels, channels * 4, norm=norm, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])

        if pool == "mean":
            self.readout = global_mean_pool 
        elif pool == "max":
            self.readout = global_max_pool 
        elif pool == "attn":
            att_net=nn.Sequential(nn.Linear(channels, channels // 2), nn.LeakyReLU(), nn.Linear(channels//2, 1))     
            self.readout = GlobalAttention(att_net)

        self.norm = norm_layer(norm, channels)
        self.prediction = nn.Linear(channels, n_classes)

    def forward(self, inputs, return_edge=False, return_atten = False):
        # x = self.stem(inputs) + self.pos_embed 
        x = self.stem(inputs) # x:(B, N, D)
        
        edge_indexs = []
        for i in range(self.n_blocks):
            x, edge_index = self.backbone[i](x)# x:(B, N, D) edge_index:# (2, B, N, k)
            edge_indexs.append(edge_index[0]) # (B, N, k)

        if return_atten:
            atten_score = self.readout.gate_nn(x)
        x = self.readout(x.squeeze(0), batch=None)
        x = self.norm(x)
        x = self.prediction(x)

        if return_edge and return_atten:
            return x, edge_indexs, atten_score
        elif return_atten:
            return x, atten_score
        elif return_edge:
            return x, edge_indexs
        else:
            return x 
        

if __name__ == "__main__":
    data = torch.randn((1, 30000, 384)).cuda()
    
    class OptInit:
        def __init__(self, dim_in=384, num_classes=2, drop_path_rate=0.0, num_knn=9, pool="attn", **kwargs):
            self.dim_in = dim_in 
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'attention' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'layer' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 1 # number of basic blocks in the backbone
            self.n_filters = 192 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.use_dilation = False # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.dropout = 0.
            self.pool =pool
            self.knn = "gateAtten"
            self.gate= True

    opt = OptInit()
    model = DeepGCN(opt).cuda()

    output = model(data) # (1,2)
    print(output.shape)
    output, edge_indexs = model(data, return_edge=True) # (1,2), [n_blocks,1,30000,k]
    print(output.shape, edge_indexs[0].shape)
    output, atten_score = model(data, return_atten=True) # (1,2), (1,30000,1)
    print(output.shape, atten_score.shape)
    output, edge_indexs, atten_score = model(data, return_edge=True, return_atten=True) # (1,2), [n_blocks,1,30000,k]
    print(output.shape, edge_indexs[0].shape, atten_score.shape)

