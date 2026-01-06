import torch
import torch.nn as nn
import torch.nn.functional as F
from model.my_modules import ResBlock
from model.HF_guided import ProgressiveFusionNet
import torch.nn.init as init


import numbers
from einops import rearrange

class AdvancedSpatialASL(nn.Module):
    def __init__(self, in_channels, kernel_size=7, reduction_ratio=4):
        super(AdvancedSpatialASL, self).__init__()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, reduction_ratio, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(reduction_ratio, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            # nn.Sigmoid()
        )

        self.weight = nn.Parameter(torch.randn(in_channels, in_channels, 3, 3))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_feat = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.spatial_attention(spatial_feat)  # [b, 1, h, w]

        modulated_x = x * spatial_attention
        out = F.conv2d(modulated_x, self.weight, padding=1)

        return out

class SpaProcess(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.post = AdvancedSpatialASL(dim)
    def forward(self, x): return self.post(x)

class SpeProcess(nn.Module):
    def __init__(self,in_channels,reduction_ratio = 8):
        super(SpeProcess,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP层
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            # nn.ReLU(),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化和最大池化的输出
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        # 将两个输出相加
        out = avg_out + max_out
        # return self.sigmoid(out)
        return out


def modulated_conv2d(
        x,  # Input tensor: [batch_size, in_channels, in_height, in_width]
        w,  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
        s,  # Style tensor: [batch_size, in_channels, 1, 1] b in 1 1
        padding=0,  # Padding: int or [padH, padW]
        bias=None,
        stride=1,
        dilation=1
):
    batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape

    # Modulate weights.
    w = w.unsqueeze(0)  # [NOIkk] 1 out in k k
    w = (w * s.unsqueeze(1))  # [NOIkk] b out in k k
    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[2:]) # 1 b*in h w
    w = w.reshape(-1, in_channels, kh, kw)   #  b*out in k k

    x = torch.nn.functional.conv2d(input=x, weight=w.to(x.dtype), bias=bias, stride=stride, padding=padding,
                                   dilation=dilation, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])

    return x    # b in h w


class ASL(nn.Module): # Adaptive Spectral Learning
    def __init__(self,in_channels):
        super(ASL,self).__init__()
        self.spe = SpeProcess(in_channels)
        self.weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False).weight # (in_channels, in_channels, 1, 1)
    def forward(self,x):
        spe_feat = self.spe(x) # (b, in_channels, 1, 1)
        m_spe = modulated_conv2d(x,self.weight,spe_feat)
        return m_spe

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.up_sample = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 16, 3, 1, 1, bias=False),
            nn.PixelShuffle(4)
        )

    def forward(self, x):
        return self.up_sample(x)

class GateNetwork(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super(GateNetwork, self).__init__()
        self.gap = nn.AdaptiveMaxPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)
        self.input_size = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.fc0 = nn.Linear(dim,num_experts)
        self.fc1 = nn.Linear(dim, num_experts)
        self.relu1 = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
        init.zeros_(self.fc1.weight)
        self.sp = nn.Softplus()
    def forward(self, x):
        # Flatten the input tensor
        x = self.gap(x)+self.gap2(x)  # b dim 1 1
        x = x.view(-1, self.input_size) #   b*1*1 dim
        inp = x
        # Pass the input through the gate network layers
        x = self.fc1(x)
        x= self.relu1(x)
        noise = self.sp(self.fc0(inp))
        noise_mean = torch.mean(noise,dim=1)
        noise_mean = noise_mean.view(-1,1)
        std = torch.std(noise,dim=1)
        std = std.view(-1,1)
        noram_noise = (noise-noise_mean)/std
        # Apply topK operation to get the highest K values and indices along dimension 1 (columns)
        topk_values, topk_indices = torch.topk(x+noram_noise, k=self.top_k, dim=1)

        # Set all non-topK values to -inf to ensure they are not selected by softmax
        mask = torch.zeros_like(x).scatter_(dim=1, index=topk_indices, value=1.0)
        x[~mask.bool()] = float('-inf')

        # Pass the masked tensor through softmax to get gating coefficients for each expert network
        gating_coeffs = self.softmax(x)

        return gating_coeffs

class SpaExpert(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super(SpaExpert, self).__init__()
        self.num_experts = num_experts
        self.gate = GateNetwork(dim, num_experts, top_k)
        self.spa_list = nn.ModuleList(
            [SpaProcess(dim) for _ in range(num_experts)]
        )

    def forward(self, x):
        # print('spa exp')
        cof = self.gate(x)
        # print("spa_cof",cof)
        out = torch.zeros_like(x).to(x.device)
        all_cof_k = torch.zeros_like(cof)  # (16, num_experts)

        expert_outputs = {}
        for idx in range(self.num_experts):
            # print(cof.dtype, cof.requires_grad, cof[:, idx])
            # if cof[:, idx].all() == 0:
            if not torch.any(cof[:, idx] > 0):
                continue
            mask = torch.where(cof[:, idx] > 0)[0]
            expert_layer = self.spa_list[idx]
            # print('ms.shape,x[mask].shape,pan.shape',ms.shape,x[mask].shape,pan.shape)
            expert_out = expert_layer(x[mask])
            expert_outputs[idx] = expert_out
            cof_k = cof[mask, idx].view(-1, 1, 1, 1)
            out[mask] += expert_out * cof_k
            all_cof_k[mask, idx] = cof[mask, idx]
        # print("spa_all_cof", all_cof_k)
        return out, all_cof_k, expert_outputs

class SpeExpert(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super(SpeExpert, self).__init__()
        self.num_experts = num_experts
        self.gate = GateNetwork(dim, num_experts, top_k)
        self.spe_list = nn.ModuleList(
            [ASL(dim) for _ in range(num_experts)]
        )

    def forward(self, x):
        # print('spe exp')
        cof = self.gate(x)
        # print("spe_cof",cof)
        out = torch.zeros_like(x).to(x.device)
        all_cof_k = torch.zeros_like(cof)  # (16, num_experts)
        expert_outputs = {}
        for idx in range(self.num_experts):
            if not torch.any(cof[:, idx] > 0):
                continue
            mask = torch.where(cof[:, idx] > 0)[0]
            # print("mask, idx", mask, idx)
            expert_layer = self.spe_list[idx]
            expert_out = expert_layer(x[mask])
            expert_outputs[idx] = expert_out
            cof_k = cof[mask, idx].view(-1, 1, 1, 1)
            out[mask] += expert_out * cof_k
            # print("cof_k", cof_k, cof_k.shape)
            all_cof_k[mask, idx] = cof[mask, idx]
        # print("spe_all_cof",all_cof_k)
        return out, all_cof_k, expert_outputs

class Net(nn.Module):
    def __init__(self,
                 dim,
                 num_experts=8,
                 num_heads = 2,
                 win_size = 4,
                 ffn_expansion_factor = 2,
                 bias = False,
                 LayerNorm_type = 'BiasFree'
                 ):
        super().__init__()
        ms_dim = 8
        self.up_sample = Upsample(ms_dim)

        self.pan_pre_exp = nn.Sequential(
            nn.Conv2d(1,dim,3,1,1,bias=False),
            nn.LeakyReLU(),
            ResBlock(dim,dim//2,dim)
        )
        self.ms_pre_exp = nn.Sequential(
            nn.Conv2d(ms_dim, dim, 3, 1, 1, bias=False),
            nn.LeakyReLU(),
            ResBlock(dim, dim // 2, dim)
        )

        self.spa_exp = SpaExpert(dim, num_experts, 4)
        self.spe_exp = SpeExpert(dim, num_experts, 4)

        self.merge = nn.Sequential(
            nn.Conv2d(dim*2,dim,1,1,0,groups=dim),
            nn.LeakyReLU(0.1)
        )
        self.process = ProgressiveFusionNet(dim, dim, num_heads=num_heads, win_size=win_size, ffn_expansion_factor=ffn_expansion_factor, LayerNorm_type=LayerNorm_type)
        self.post = nn.Conv2d(dim,ms_dim,1,1,0)

    def forward(self, ms, pan):
        output_list = []

        ms_up = self.up_sample(ms)

        pan_pre_exp = self.pan_pre_exp(pan)
        ms_pre_exp = self.ms_pre_exp(ms_up)

        spa, spa_cofk,_ = self.spa_exp(self.merge(torch.cat([pan_pre_exp,ms_pre_exp],dim=1)))
        spe, spe_cofk,_ = self.spe_exp(self.merge(torch.cat([pan_pre_exp,ms_pre_exp],dim=1)))


        process = self.process(spa,spe,pan,ms_up)

        output = self.post(process)

        output_list.append(output)

        return output_list[-1] + ms_up, spa_cofk, spe_cofk




