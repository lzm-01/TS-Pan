import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  
from timm.models.layers import to_2tuple, trunc_normal_
import numbers
from model.CTB import ChannelTransformerBlock_C
from model.q_shift_style_rwkv import VRWKV_SpatialMix_Diag

class Down(nn.Module):
    def __init__(self, n_feat):
        super(Down, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2,1,1,0),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Up(nn.Module):
    def __init__(self, n_feat):
        super(Up, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2,1,1,0),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)




def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class PointConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(x)

class SSAIB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.dwconv2 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.pwconv = nn.Conv2d(dim * 2, dim, 1)
        self.gate_conv = nn.Conv2d(dim * 2, dim, 1)
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, spa, spe):
        spa_feat = self.act(self.norm1(self.dwconv1(spa)))
        spe_feat = self.act(self.norm2(self.dwconv2(spe)))

        gate_input = torch.cat([spa_feat, spe_feat], dim=1)
        gate = torch.sigmoid(self.gate_conv(gate_input))

        fused_feat = self.pwconv(gate_input)
        output = fused_feat * gate + (spa + spe) * (1 - gate)

        return self.dropout(self.act(output))

class WM_CA(nn.Module):
    """
    High-Frequency guided Window-based Multi-Head Cross-Attention.
    """

    def __init__(self, channels, win_size, num_heads):
        super(WM_CA, self).__init__()

        self.channels = channels
        self.win_size = win_size
        self.num_heads = num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.head_channels = self.channels // self.num_heads
        self.scale = self.head_channels ** -0.5

        self.q_generator = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels)
        )

        self.kv_generator = nn.Sequential(
            nn.Conv2d(self.channels, self.channels * 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(self.channels * 2, self.channels * 2, kernel_size=3, stride=1, padding=1,
                      groups=self.channels * 2)
        )
   
        self.softmax = nn.Softmax(dim=-1)
        self.proj_out = PointConv(self.channels, self.channels)

        seq_l = self.win_size * self.win_size
        self.pos_emb_q_k = nn.Parameter(torch.Tensor(1, self.num_heads, seq_l, seq_l))
        trunc_normal_(self.pos_emb_q_k, std=.02)  # Initialize with a smaller std deviation, a common practice

    def _windowize_bchw(self, img_tensor_bchw):
        """
        Converts (B, C, H, W) to (B, C, NumWindows, TokensPerWindow)
        """
        windowed_tensor = rearrange(img_tensor_bchw, 'b c (h i) (w j) -> b c (h w) (i j)',
                                    i=self.win_size, j=self.win_size)
        return windowed_tensor

    def _reshape_for_multihead_bchw(self, tensor_qkv_bcnwinsl):
        """
        Reshapes Q, K, V for multi-head attention.
        Input: (B, C, NumWindows, SeqLen)
        Output: (B * NumWindows, NumHeads, SeqLen, HeadChannels)
        """
        multihead_tensor = rearrange(tensor_qkv_bcnwinsl, 'b (nh ch) nw sl -> (b nw) nh sl ch',
                                     nh=self.num_heads)
        return multihead_tensor

    def _unwindowize_bchw(self, windowed_output_bcnwinsl, H_orig, W_orig):
        """
        Converts (B, C, NumWindows, TokensPerWindow) back to (B, C, H, W).
        """
        num_windows_h = H_orig // self.win_size
        num_windows_w = W_orig // self.win_size
        img_like_tensor = rearrange(windowed_output_bcnwinsl, 'b c (h w) (i j) -> b c (h i) (w j)',
                                    h=num_windows_h, w=num_windows_w,
                                    i=self.win_size, j=self.win_size)
        return img_like_tensor

    def forward(self, x_bchw, y_bchw):
        # x_bchw: Input for Key and Value, shape (B, C, H, W) - e.g., MS features
        # y_bchw: Input for Query, shape (B, C, H, W) - e.g., PAN features (HF-guided)

        B_orig, C_orig, H_orig, W_orig = x_bchw.shape
        assert C_orig == self.channels, f"Input x channels mismatch: expected {self.channels}, got {C_orig}"
        assert y_bchw.shape == x_bchw.shape, "x and y must have the same shape"
        assert H_orig % self.win_size == 0 and W_orig % self.win_size == 0, \
            "Height and Width must be divisible by window size"

        q_feat = self.q_generator(y_bchw)
        kv_feat = self.kv_generator(x_bchw)
        k_feat, v_feat = kv_feat.chunk(2, dim=1)

        q_win = self._windowize_bchw(q_feat)
        k_win = self._windowize_bchw(k_feat)
        v_win = self._windowize_bchw(v_feat)

        q = self._reshape_for_multihead_bchw(q_win)
        k = self._reshape_for_multihead_bchw(k_win)
        v = self._reshape_for_multihead_bchw(v_win)

        q = q * self.scale
        sim = torch.einsum('b h i c, b h j c -> b h i j', q, k)
        sim = sim + self.pos_emb_q_k
        attention_map = self.softmax(sim)

        out_multihead = torch.einsum('b h i j, b h j c -> b h i c', attention_map, v)

        out_concat_heads = rearrange(out_multihead, '(b nw) nh sl ch -> b (nh ch) nw sl',
                                     b=B_orig)

        out_projected = self.proj_out(out_concat_heads)
        output_bchw = self._unwindowize_bchw(out_projected, H_orig, W_orig)

        return output_bchw

class Freprocess(nn.Module):
    def __init__(self, channels):
        super(Freprocess, self).__init__()
        self.pre1 = nn.Conv2d(channels,channels,1,1,0)
        self.pre2 = nn.Conv2d(channels,channels,1,1,0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, msf, panf):

        _, _, H, W = msf.shape
        msF = torch.fft.rfft2(self.pre1(msf)+1e-8, norm='backward')
        panF = torch.fft.rfft2(self.pre2(panf)+1e-8, norm='backward')
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        panF_amp = torch.abs(panF)
        panF_pha = torch.angle(panF)
        amp_fuse = self.amp_fuse(torch.cat([msF_amp,panF_amp],1))
        pha_fuse = self.pha_fuse(torch.cat([msF_pha,panF_pha],1))

        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

        return self.post(out)


class S2_HF_SPA(nn.Module):
    def __init__(self,channels, win_size, num_heads):
        super(S2_HF_SPA, self).__init__()

        self.channels = channels
        self.half_channels = channels // 2
        self.win_size = win_size
        self.num_heads = num_heads
        self.HF_1 = Freprocess(self.half_channels)
        self.HF_2 = WM_CA(self.half_channels, win_size, num_heads)
        self.q_shift_spatial_rwkv = VRWKV_SpatialMix_Diag(self.channels)
        self.after_spa_norm = nn.LayerNorm(self.channels)

        self.tail = nn.Sequential(
            PointConv(self.channels, self.channels),
            nn.GELU(),
            PointConv(self.channels, self.channels, groups=self.channels)  ### depth_wise
        )
    def forward(self, s2, hf):
        b, c, h, w = s2.shape
        resolution = (h, w)

        s2_half_1, s2_half_2 = torch.chunk(s2, 2, dim=1)
        hf_half_1, hf_half_2 = torch.chunk(hf, 2, dim=1) ### B C//2 H W

        fre = self.HF_1(s2_half_1,hf_half_1)
        hgb = self.HF_2(s2_half_2,hf_half_2) ### B C//2 H W
        local_global = self.tail(torch.cat([fre, hgb], dim=1)) ### B C H W

        local_global_flat =  rearrange(local_global, 'b c h w -> b (h w) c').contiguous()
        local_global_flat = self.q_shift_spatial_rwkv(local_global_flat,resolution)
        local_global_un = rearrange(self.after_spa_norm(local_global_flat), 'b (h w) c -> b c h w', h=h, w=w).contiguous()

        return local_global_un + local_global

class CMX_Diag(nn.Module):
    def __init__(self, channels):
        super(CMX_Diag, self).__init__()
        self.channels = channels

        self.channel_mix_diag = VRWKV_ChannelMix_Diag(self.channels)

    def forward(self, x):
        b, c, h, w = x.shape

        resolution = (h, w)

        x_flat = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        x_flat = self.channel_mix_diag(x_flat, resolution)
        x_unfold = rearrange(x_flat, 'b (h w) c -> b c h w', h=h, w=w).contiguous()

        return x_unfold

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = PointConv(dim, hidden_features * 2)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = PointConv(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class HighFre_Guided_Transformer(nn.Module):
    def __init__(self, dim, num_heads,  win_size, ffn_expansion_factor, bias, LayerNorm_type):
        super(HighFre_Guided_Transformer, self).__init__()

        self.norm1_s2 = LayerNorm(dim, LayerNorm_type)
        self.norm1_hf = LayerNorm(dim, LayerNorm_type)
        self.attn = S2_HF_SPA(dim, win_size, num_heads)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        # self.ffn = CMX_Diag(dim)
    def forward(self, s2, hf):
        # print("s2.shape", s2.shape)
        x = s2 + self.attn(self.norm1_s2(s2),self.norm1_hf(hf))
        x = x + self.ffn(self.norm2(x))

        return x





class HFG_Block_T(nn.Module):
    def __init__(self, dim, num_heads, win_size,ffn_expansion_factor, bias, LayerNorm_type):
        super(HFG_Block_T, self).__init__()

        self.dim = dim
        self.bottleneck_dim = dim * 2
        self.num_heads = num_heads
        self.win_size = win_size
        self.down_D = Down(dim)
        self.down_ = Down(dim)
        # self.up_D = Up(dim*2)
        self.up_ = Up(dim*2)

        self.conv_concat = PointConv(dim*2, dim)

        self.encoder = HighFre_Guided_Transformer(self.dim, self.num_heads, self.win_size, ffn_expansion_factor, bias, LayerNorm_type)
        self.bottleneck = HighFre_Guided_Transformer(self.bottleneck_dim, self.num_heads*2,  self.win_size, ffn_expansion_factor, bias, LayerNorm_type)
        self.decoder = HighFre_Guided_Transformer(self.dim, self.num_heads, self.win_size, ffn_expansion_factor, bias, LayerNorm_type)

    def forward(self, s2, hf):
        en = self.encoder(s2, hf)
        en_down = self.down_(en)
        hf_down = self.down_D(hf)
        bottleneck = self.bottleneck(en_down, hf_down)
        bottleneck_up = self.up_(bottleneck)
        # hf_up = self.up_D(hf_down)
        de = self.decoder(self.conv_concat(torch.cat([bottleneck_up, en],dim=1)), hf)
        return de






