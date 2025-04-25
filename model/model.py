import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from function.LGFA import LGFA
from function.NRAttention import NRAttention



def GaussProjection(x, mean, std):
    sigma = math.sqrt(2 * math.pi) * std
    x_out = torch.exp(-(x - mean) ** 2 / (2 * std ** 2)) / sigma
    return x_out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels=32, reduction_ratio=2, pool_types=['avg']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        # Gauss modulation
        mean = torch.mean(channel_att_sum).detach()
        std = torch.std(channel_att_sum).detach()
        scale = GaussProjection(channel_att_sum, mean, std).unsqueeze(2).unsqueeze(3).expand_as(x)

        # scale = scale / torch.max(scale)
        return (x * scale).permute(0, 3, 2, 1)


import os
import sys
import copy
import math
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from centroid import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GroupAttention(nn.Module):
    """
    LSA: self attention within a group
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=1):
        assert ws != 1
        super(GroupAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, H, W):
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        qkv = self.qkv(x).reshape(B, total_groups, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        # B, hw, ws*ws, 3, n_head, head_dim -> 3, B, hw, n_head, ws*ws, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(
            attn)  # attn @ v-> B, hw, n_head, ws*ws, head_dim -> (t(2,3)) B, hw, ws*ws, n_head,  head_dim
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GroupBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=8):
        super().__init__()
        #         del self.attn
        if ws == 1:
            self.attn = torch.Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)
        else:
            self.attn = GroupAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, ws)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ws = ws

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, d_model=64):
        super(Transformer, self).__init__()

        self.GroupBlock = GroupBlock(d_model, 8, ws=10)

    #         self.pe = PositionalEncodingLearned1D(d_model)

    def forward(self, x, H, W, d):
        for i in range(d):
            x = self.GroupBlock(x, H, W)
        x = x.permute(0, 2, 1)

        return x


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, n_pts, k)
    return idx


def get_graph_feature(coor, nor, k=20, idx=None):
    # coor:(B, 3, N)
    # nor:(B, 3, N)
    batch_size, num_dims_c, ncells = coor.shape
    _, num_dims_n, _ = coor.shape
    coor = coor.view(batch_size, -1, ncells)
    if idx is None:
        idx = knn(coor, k=k)  # (B, N, k)
        idx_r = idx

    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     device = torch.device('cpu')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * ncells  # (B, 1, 1)
    idx = idx + idx_base  # (B, N, k)
    idx = idx.view(-1)  # (B*N*k, )

    coor = coor.transpose(2, 1).contiguous()  # (B, N, 3)
    nor = nor.transpose(2, 1).contiguous()  # (B, N, 3)

    # coor
    coor_feature = coor.view(batch_size * ncells, -1)[idx, :]
    coor_feature = coor_feature.view(batch_size, ncells, k, num_dims_c)
    coor = coor.view(batch_size, ncells, 1, num_dims_c).repeat(1, 1, k, 1)
    coor_feature = torch.cat((coor_feature, coor), dim=3).permute(0, 3, 1, 2).contiguous()

    # normal vector
    nor_feature = nor.view(batch_size * ncells, -1)[idx, :]
    nor_feature = nor_feature.view(batch_size, ncells, k, num_dims_n)
    nor = nor.view(batch_size, ncells, 1, num_dims_n).repeat(1, 1, k, 1)
    nor_feature = torch.cat((nor_feature, nor), dim=3).permute(0, 3, 1, 2).contiguous()

    return coor_feature, nor_feature, idx_r  # (B, 2*3, N, k)


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class AttentionFusion(nn.Module):
    def __init__(self, point_dim, centroid_dim):
        super(AttentionFusion, self).__init__()
        self.point_transform = nn.Linear(point_dim, point_dim)
        self.centroid_transform = nn.Linear(centroid_dim, point_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, point_features, centroid_features):
        """
        :param point_features: (B, N, C_p) 点特征
        :param centroid_features: (B, num_classes, C_c) 质心特征
        :return: fused_features (B, N, C_p)
        """
        # 线性变换
        point_features_proj = self.point_transform(point_features)  # (B, N, C_p)
        centroid_features_proj = self.centroid_transform(centroid_features)  # (B, num_classes, C_p)

        # 计算注意力权重
        attention_scores = torch.bmm(point_features_proj, centroid_features_proj.transpose(1, 2))  # (B, N, num_classes)
        attention_weights = self.softmax(attention_scores)  # (B, N, num_classes)

        # 通过权重加权质心特征
        centroid_features_weighted = torch.bmm(attention_weights, centroid_features)  # (B, N, C_c)

        # 融合点特征与加权质心特征
        fused_features = point_features + centroid_features_weighted  # (B, N, C_p)

        return fused_features

class CombinedFusion(nn.Module):
    def __init__(self, point_dim, centroid_dim, temperature=1.0):
        super(CombinedFusion, self).__init__()
        self.point_transform = nn.Linear(point_dim, point_dim)
        self.centroid_transform = nn.Linear(centroid_dim, point_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.temperature = temperature

    def forward(self, point_coords, point_features, centroid_coords, centroid_features):
        """
        :param point_coords: (B, N, 3) 点的坐标
        :param point_features: (B, N, C_p) 点的特征
        :param centroid_coords: (B, num_classes, 3) 质心的坐标
        :param centroid_features: (B, num_classes, C_c) 质心的特征
        :return: fused_features (B, N, C_p)
        """
        # --- 注意力权重计算 ---
        point_features_proj = self.point_transform(point_features)  # (B, N, C_p)
        centroid_features_proj = self.centroid_transform(centroid_features)  # (B, num_classes, C_p)
        attention_scores = torch.bmm(point_features_proj, centroid_features_proj.transpose(1, 2))  # (B, N, num_classes)
        attention_weights = self.softmax(attention_scores)  # (B, N, num_classes)

        # --- 距离权重计算 ---
        distances = torch.cdist(point_coords, centroid_coords, p=2)  # (B, N, num_classes)
        distance_weights = torch.exp(-distances / self.temperature)  # 高斯距离权重
        distance_weights = distance_weights / distance_weights.sum(dim=-1, keepdim=True)  # 归一化 (B, N, num_classes)

        # --- 综合权重计算 ---
        # combined_weights = attention_weights * distance_weights  # 综合权重 (B, N, num_classes)
        combined_weights = attention_weights  # 综合权重 (B, N, num_classes)

        # --- 检查并扩展 centroid_features 的维度 ---
        if centroid_features.shape[1] == 1:  # 如果质心数量为1
            centroid_features = centroid_features.expand(-1, combined_weights.shape[-1], -1)  # 扩展为 (B, num_classes, C_c)

        # --- 加权质心特征 ---
        weighted_centroid_features = torch.bmm(combined_weights, centroid_features)  # (B, N, C_c)

        # --- 融合点特征与质心特征 ---
        fused_features = point_features + weighted_centroid_features  # (B, N, C_p)

        return fused_features


class CTAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=2):
        super(CTAM, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x: (B, C, N) 输入点云特征
        :return: (B, C, N) 加权后的特征
        """
        B, C, N = x.size()

        # 全局池化
        global_context = x.mean(dim=-1)  # (B, C)

        # 通道权重生成
        weights = self.mlp(global_context)  # (B, C)
        weights = weights.unsqueeze(-1)  # (B, C, 1)

        return x * weights



class My_Seg(nn.Module):
    def __init__(self, num_classes=15, num_neighbor=20):
        super(My_Seg, self).__init__()

        self.k = num_neighbor

        self.stn_c1 = STNkd(k=3)
        self.bn1_c = nn.BatchNorm2d(64)

        self.bn2_c = nn.BatchNorm2d(128)
        self.bn3_c = nn.BatchNorm2d(256)
        self.bn4_c = nn.BatchNorm2d(256)
        self.bn5_c = nn.BatchNorm1d(256)
        self.conv1_c = nn.Sequential(nn.Conv2d(3 * 2, 64, kernel_size=1, bias=False),
                                     self.bn1_c,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv2_c = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                     self.bn2_c,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv3_c = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                     self.bn3_c,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv5_c = nn.Sequential(nn.Conv1d(448, 256, kernel_size=1, bias=False),
                                     self.bn5_c,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.stn_n1 = STNkd(k=3)
        self.bn1_n = nn.BatchNorm2d(64)

        self.bn2_n = nn.BatchNorm2d(128)
        self.bn3_n = nn.BatchNorm2d(256)
        self.bn4_n = nn.BatchNorm2d(256)
        self.bn5_n = nn.BatchNorm1d(256)
        self.conv1_n = nn.Sequential(nn.Conv2d(3 * 2, 64, kernel_size=1, bias=False),
                                     self.bn1_n,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv2_n = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                     self.bn2_n,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv3_n = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                     self.bn3_n,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv5_n = nn.Sequential(nn.Conv1d(448, 256, kernel_size=1, bias=False),
                                     self.bn5_n,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.pred1 = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(512),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pred2 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pred3 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.6)
        self.dp2 = nn.Dropout(p=0.6)
        self.dp3 = nn.Dropout(p=0.6)
        self.pred4 = nn.Sequential(nn.Conv1d(128, num_classes, kernel_size=1, bias=False))

        self.nra1 = NRAttention(in_dim = 64)
        self.nra2 = NRAttention(in_dim = 128)
        self.nra3 = NRAttention(in_dim = 256)

        self.attention_fusion1 = AttentionFusion(point_dim=64, centroid_dim=64)
        self.attention_fusion2 = AttentionFusion(point_dim=128, centroid_dim=128)
        self.attention_fusion3 = AttentionFusion(point_dim=256, centroid_dim=256)

        self.combined_fusion1 = CombinedFusion(point_dim=64, centroid_dim=64, temperature=0.1)
        self.combined_fusion2 = CombinedFusion(point_dim=128, centroid_dim=128, temperature=0.1)
        self.combined_fusion3 = CombinedFusion(point_dim=256, centroid_dim=256, temperature=0.1)

        self.LGFA1 = LGFA(in_channel=64, out_channel_list=[64], k_hat=32, bias=False)
        self.LGFA2 = LGFA(in_channel=128, out_channel_list=[128], k_hat=32, bias=False)
        self.LGFA3 = LGFA(in_channel=256, out_channel_list=[256], k_hat=32, bias=False)
        # self.transformer1 = Transformer(64)
        # self.transformer2 = Transformer(128)
        # self.transformer3 = Transformer(256)

        self.ChannelAMM = ChannelGate()
        self.ctam1 = CTAM(gate_channels=64, reduction_ratio=2)
        self.ctam2 = CTAM(gate_channels=128, reduction_ratio=2)
        self.ctam3 = CTAM(gate_channels=256, reduction_ratio=2)

        self.num_classes = num_classes

        # 第1个图
        self.FFN1_n = nn.Sequential(torch.nn.Conv1d(64, 64 * 2, 1), nn.LeakyReLU(negative_slope=0.2),
                                    nn.Dropout(0.1), torch.nn.Conv1d(64 * 2, 64, 1))
        self.res_linear1_1n = torch.nn.Conv1d(3, 64, 1)
        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = nn.BatchNorm1d(64)

        # 第2个图
        self.FFN2_n = nn.Sequential(torch.nn.Conv1d(128, 128 * 2, 1), nn.LeakyReLU(negative_slope=0.2),
                                    nn.Dropout(0.1), torch.nn.Conv1d(128 * 2, 128, 1))
        self.res_linear2_1n = torch.nn.Conv1d(64, 128, 1)
        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = nn.BatchNorm1d(128)

        # 第3个图
        self.FFN3_n = nn.Sequential(torch.nn.Conv1d(256, 256 * 2, 1), nn.LeakyReLU(negative_slope=0.2),
                                    nn.Dropout(0.1), torch.nn.Conv1d(256 * 2, 256, 1))
        self.res_linear3_1n = torch.nn.Conv1d(128, 256, 1)
        self.dropout3 = nn.Dropout(0.1)
        self.norm3 = nn.BatchNorm1d(256)

        self.centroid = centroid()
        self.pe1 = nn.Sequential(nn.Linear(42, 64), nn.ReLU())
        self.pe2 = nn.Sequential(nn.Linear(42, 128), nn.ReLU())
        self.pe3 = nn.Sequential(nn.Linear(42, 256), nn.ReLU())

    def forward(self, x):
        # 输入的形状  x:(B, 6, N)
        # 获取点坐标信息coor， 法向量nor
        coor = x[:, :3, :]  # (B, 3, N)  (1, 3, 10000)
        nor = x[:, 3:, :]  # (B, 3, N)   (1, 3, 10000)
        coor_raw = coor
        batch_size = x.size(0)  # 求 B 这里B就是train中的batch
        ncells = x.size(2)  # N  N为牙模型的点数10000

        cent0, label, weight = self.centroid(x)  # 预测每个标签的质心点
        cent = cent0.view(batch_size, -1)  # cent = (1, 42)
        # cent0 = (1, 3, 14)  label = (1, 14)  weight = (1, 3)
        #         print(weight.shape)

        # coor input transform
        # STNkd 模块通过学习变换矩阵来增强模型对输入数据的处理能力，尤其是在面对各种几何变换时，提高了模型的鲁棒性和泛化能力。
        input_trans_c = self.stn_c1(coor)  # (B, 3, 3) 一个空间变换矩阵，用于对输入数据进行几何变换
        coor = coor.transpose(2, 1)  # (B, 3, N) -> (B, N, 3)
        coor = torch.bmm(coor, input_trans_c)  # torch.bmm 是一个用于执行批量矩阵乘法的函数。
        coor = coor.transpose(2, 1)  # (B, N, 3) -> (B, 3, N)

        # nor input transform
        input_trans_n = self.stn_n1(nor)  # (B, 3, 3)
        nor = nor.transpose(2, 1)  # (B, 3, N) -> (B, N, 3)
        nor = torch.bmm(nor, input_trans_n)
        nor = nor.transpose(2, 1)  # (B, N, 3) -> (B, 3, N)
        #         print("coor, nor:",coor.shape,nor.shape)

        # get_graph_feature从三维坐标和法向量中提取局部特征
        # 通过 KNN方法和特征拼接，函数为每个点生成了局部特征表示，
        coor1, nor1, idx = get_graph_feature(coor, nor, k=self.k)  # (B, 3, N) -> (B, 3*2, N, k)
        coor1 = self.conv1_c(coor1)  # (B, 3*2, N, k) -> (B, 64, N, k)
        coor1 = coor1.max(dim=-1, keepdim=False)[0]  # (B, 64, N, k) -> (B, 64, N)
        # 第一级特征融合
        coor1 = coor1.permute(0, 2, 1)  # (B, 64, N) -> (B, N, 64)
        cent1 = self.pe1(cent).unsqueeze(1)  # (B, 1, 64)
        coor_coords = coor_raw.permute(0, 2, 1)  # (B, N, 3)
        cent_coords = cent0.permute(0, 2, 1)  # (B, num_classes, 3)
        coor1 = self.combined_fusion1(coor_coords, coor1, cent_coords, cent1)  # 动态距离权重融合
        # coor1 = coor1 + cent1
        coor1 = coor1.permute(0, 2, 1)  # 转回 (B, 64, N)
        coor1 = self.nra1(coor1)
        coor1 = self.LGFA1(coor1, idx)
        # coor1 = self.transformer1(coor1, 100, 100, 2) # (B, N, 64) -> (B, 64, N)

        nor1 = self.conv1_n(nor1)  # (B, 3*2, N, k) -> (B, 64, N, k)
        nor1 = self.ChannelAMM(nor1)  # (B, 64, N, k)
        nor1 = nor1.max(dim=-1, keepdim=False)[0]  # (B, 64, N, k) -> (B, 64, N)
        nor1 = self.ctam1(nor1)

        coor2, nor2, idx = get_graph_feature(coor1, nor1, k=self.k)  # (B, 64, N) -> (B, 64*2, N, k)
        coor2 = self.conv2_c(coor2)  # (B, 64*2, N, k) -> (B, 128, N, k)
        coor2 = coor2.max(dim=-1, keepdim=False)[0]  # (B, 128, N, k) -> (B, 128, N)
        # 第二级特征融合
        coor2 = coor2.permute(0, 2, 1)  # (B, 128, N) -> (B, N, 128)
        cent2 = self.pe2(cent).unsqueeze(1)  # (B, 1, 128)
        coor2 = self.combined_fusion2(coor_coords, coor2, cent_coords, cent2)  # 动态距离权重融合
        # coor2 = coor2 + cent2
        coor2 = coor2.permute(0, 2, 1)  # 转回 (B, 128, N)
        # coor2 = self.transformer2(coor2, 100, 100, 3)# (B, N, 128) -> (B, 128, N)
        coor2 = self.nra2(coor2)
        coor2 = self.LGFA2(coor2, idx)

        nor2 = self.conv2_n(nor2)  # (B, 64*2, N, k) -> (B, 128, N, k)
        nor2 = self.ChannelAMM(nor2)  # (B, 128, N, k)
        nor2 = nor2.max(dim=-1, keepdim=False)[0]  # (B, 128, N, k) -> (B, 128, N)
        nor2 = self.ctam2(nor2)

        coor3, nor3, idx = get_graph_feature(nor2, coor2, k=self.k)  # (B, 64, N) -> (B, 64*2, N, k)
        coor3 = self.conv3_c(coor3)  # (B, 64*2, N, k) -> (B, 256, N, k)
        coor3 = coor3.max(dim=-1, keepdim=False)[0]  # (B, 256, N, k) -> (B, 256, N)

        # 第三级特征融合
        coor3 = coor3.permute(0, 2, 1)  # (B, 256, N) -> (B, N, 256)
        cent3 = self.pe3(cent).unsqueeze(1)  # (B, 1, 256)
        coor3 = self.combined_fusion3(coor_coords, coor3, cent_coords, cent3)  # 动态距离权重融合
        # coor3 = coor3 + cent3
        coor3 = coor3.permute(0, 2, 1)  # 转回 (B, 256, N)
        # coor3 = self.transformer3(coor3, 100, 100, 2)
        coor3 = self.nra3(coor3)
        coor3 = self.LGFA3(coor3, idx)

        nor3 = self.conv3_n(nor3)  # (B, 64*2, N, k) -> (B, 256, N, k)
        nor3 = self.ChannelAMM(nor3)
        nor3 = nor3.max(dim=-1, keepdim=False)[0]  # (B, 128, N, k) -> (B, 128, N)
        nor3 = self.ctam3(nor3)

        coor = torch.cat((coor1, coor2, coor3), dim=1)  # (B, 64+128+256, N)   448
        coor = self.conv5_c(coor)  # (B, 448, N) -> (B, 256, N)
        nor = torch.cat((nor1, nor2, nor3), dim=1)  # (B, 64+128+256, N)
        nor = self.conv5_n(nor)  # (B, 448, N) -> (B, 256, N)

        x = torch.cat((coor, nor), dim=1)  # (B, 256*2, N)

        x = self.pred1(x)  # (B, 256*2, N) -> (B, 512, N)
        x = self.dp1(x)
        x = self.pred2(x)  # (B, 512, N) -> (B, 256, N)
        x = self.dp2(x)
        x = self.pred3(x)  # (B, 256, N) -> (B, 128, N)
        x = self.dp3(x)

        x = self.pred4(x)  # (B, 128, N) -> (B, 15, N)

        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, cent0.permute(0, 2, 1), label, weight.mean(axis=0)

# import torch
# import torchvision
# from thop import profile
# input0 = torch.rand(1, 6, 10000).cuda()
# model = My_Seg(num_classes=15, num_neighbor=32).cuda()
# out = model(input0)
# print(out[1].shape)
#
# flops, params = profile(model, (input0,))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))

# import torch
# import torchvision
# from thop import profile
# import time
# # 生成输入数据并将其放入 GPU
# input0 = torch.rand(1, 6, 10000).cuda()
# # 实例化模型并放入 GPU
# model = My_Seg(num_classes=15, num_neighbor=32).cuda()
# # 使用 time.time() 记录开始时间
# start_time = time.time()
# # 前向传播
# out = model(input0)
# # 计算总时间（包括 GPU-CPU 同步的时间）
# end_time = time.time()
# elapsed_time = end_time - start_time
# # 输出运算时间（以秒为单位）
# print(f"Forward pass time: {elapsed_time:.3f} s")
# # 输出模型的第二个输出的形状
# print(out[1].shape)
# # 计算 FLOPs 和参数量
# flops, params = profile(model, (input0,))
# print('FLOPs: ', flops, 'Params: ', params)
# print('FLOPs: %.2f G, Params: %.2f M' % (flops / 1e9, params / 1e6))
