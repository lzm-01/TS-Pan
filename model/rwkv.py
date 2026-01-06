import math, os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

T_MAX = 256*256 ### train 64*64 test 256*256 512*512

from torch.utils.cpp_extension import load

wkv_cuda = load(name="wkv", sources=["./model/cuda/wkv_op.cpp", "./model/cuda/wkv_cuda.cu"],
                verbose=True,
                extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3',
                                   f'-DTmax={T_MAX}'])


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    assert gamma <= 1/4
    B, N, C = input.shape
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
    output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    return output.flatten(2).transpose(1, 2)
def diagonal_group_shift(input_tensor, patch_resolution, channel_gamma=1 / 4, num_actions=4):

    B, N, C_in = input_tensor.shape
    H, W = patch_resolution

    x_reshaped = input_tensor.transpose(1, 2).reshape(B, C_in, H, W)

 
    gamma_c = int(C_in * channel_gamma)
    if gamma_c == 0 and C_in > 0 and channel_gamma > 0:
        gamma_c = 1  # Ensure at least 1 channel if possible
        # print(f"Info: Adjusted gamma_c to 1 for diagonal_group_shift.")

    output_shifted_parts = []
    current_c_idx = 0

    # Action 1: Down-Right Diagonal Shift
    if current_c_idx < C_in and num_actions >= 1:
        end_c_idx = min(C_in, current_c_idx + gamma_c)
        if end_c_idx > current_c_idx:
            channels_to_shift = x_reshaped[:, current_c_idx:end_c_idx, :, :]
            output_shifted_parts.append(torch.roll(channels_to_shift, shifts=(1, 1), dims=(2, 3)))
        current_c_idx = end_c_idx

    # Action 2: Up-Left Diagonal Shift
    if current_c_idx < C_in and num_actions >= 2:
        end_c_idx = min(C_in, current_c_idx + gamma_c)
        if end_c_idx > current_c_idx:
            channels_to_shift = x_reshaped[:, current_c_idx:end_c_idx, :, :]
            output_shifted_parts.append(torch.roll(channels_to_shift, shifts=(-1, -1), dims=(2, 3)))
        current_c_idx = end_c_idx

    # Action 3: Up-Right Diagonal Shift
    if current_c_idx < C_in and num_actions >= 3:
        end_c_idx = min(C_in, current_c_idx + gamma_c)
        if end_c_idx > current_c_idx:
            channels_to_shift = x_reshaped[:, current_c_idx:end_c_idx, :, :]
            output_shifted_parts.append(torch.roll(channels_to_shift, shifts=(-1, 1), dims=(2, 3)))
        current_c_idx = end_c_idx

    # Action 4: Down-Left Diagonal Shift
    if current_c_idx < C_in and num_actions >= 4:  # Or just num_actions, if it's always 4
        end_c_idx = min(C_in, current_c_idx + gamma_c)
        if end_c_idx > current_c_idx:
            channels_to_shift = x_reshaped[:, current_c_idx:end_c_idx, :, :]
            output_shifted_parts.append(torch.roll(channels_to_shift, shifts=(1, -1), dims=(2, 3)))
        current_c_idx = end_c_idx

    if not output_shifted_parts: 
        if C_in > 0:  
            xx_final_reshaped = x_reshaped  # Return original if no shifts applied
        else:
            xx_final_reshaped = torch.empty_like(x_reshaped[:, :0, :, :]) 
    else:
        xx_processed = torch.cat(output_shifted_parts, dim=1)
        if current_c_idx < C_in:
            identity_part = x_reshaped[:, current_c_idx:, :, :]
            xx_final_reshaped = torch.cat([xx_processed, identity_part], dim=1)
        else:
            xx_final_reshaped = xx_processed

    if C_in > 0 and xx_final_reshaped.shape[1] != C_in:
        pass

    return xx_final_reshaped.flatten(2).transpose(1, 2)

class VRWKV_SpatialMix_Diag(nn.Module):
    def __init__(self, n_embd, channel_gamma=1 / 4, shift_pixel=1):
        super().__init__()
        self.n_embd = n_embd
        attn_sz = n_embd
        self._init_weights()
        self.shift_pixel = shift_pixel
        if shift_pixel > 0:
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None
        self.alpha = nn.Parameter(torch.ones(1),requires_grad=True)
        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        self.key_norm = nn.LayerNorm(n_embd)
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def _init_weights(self):
        self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
        self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
        self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)

    def jit_func(self, x, patch_resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        # Use xk, xv, xr to produce k, v, r
        if self.shift_pixel > 0:
            xx_1 = q_shift(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xx_2 = diagonal_group_shift(x, patch_resolution)
            xx = self.alpha * xx_1 + (1-self.alpha) * xx_2

            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)
        return sr, k, v

    def forward(self, x, patch_resolution=None):
        B, T, C = x.size()
        sr, k, v = self.jit_func(x, patch_resolution)
        x = RUN_CUDA(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
        x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x

class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, hidden_rate=4, channel_gamma=1/4, shift_pixel=1,
                 key_norm=True):
        super().__init__()
        # self.layer_id = layer_id
        # self.n_layer = n_layer
        self.channel_gamma = channel_gamma
        self.shift_pixel = shift_pixel
        self.n_embd = n_embd
        self._init_weights()
        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)

        # self.omni_shift = OmniShift(dim=n_embd)

        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None

        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)
        self.value.scale_init = 0
        self.receptance.scale_init = 0
    def _init_weights(self):
        self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            if self.shift_pixel > 0:
                xx = q_shift(x, self.shift_pixel, self.channel_gamma, patch_resolution)
                xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
                xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
            else:
                xk = x
                xr = x

            k = self.key(xk)
            k = torch.square(torch.relu(k))
            if self.key_norm is not None:
                k = self.key_norm(k)
            kv = self.value(k)
            x = torch.sigmoid(self.receptance(xr)) * kv
            return x

        return _inner_forward(x)
