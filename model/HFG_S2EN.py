import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MHF_FE import HF_PAN, HF_MS
from model.my_modules import HFG_LGEB, HFG_CEB, Up, Down, SSAIB


class SpanConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SpanConv, self).__init__()
        self.in_planes = in_channels
        self.out_planes = out_channels
        self.kernel_size = kernel_size

        self.point_wise_1 = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=True)

        self.depth_wise_1 = nn.Conv2d(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=(kernel_size - 1) // 2,
                                    groups=out_channels,
                                    bias=True)

        self.point_wise_2 = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=True)

        self.depth_wise_2 = nn.Conv2d(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=(kernel_size - 1) // 2,
                                    groups=out_channels,
                                    bias=True)


    def forward(self, x):  #
        out_tmp_1 = self.point_wise_1(x)  #
        out_tmp_1 = self.depth_wise_1(out_tmp_1)  #

        out_tmp_2 = self.point_wise_2(x)  #
        out_tmp_2 = self.depth_wise_2(out_tmp_2)  #

        out = out_tmp_1 + out_tmp_2

        return out

class HFG_S2EN(nn.Module):
    def __init__(self,in_channels,out_channels,num_heads,  win_size, ffn_expansion_factor=2, bias=False, LayerNorm_type="BiasFree"):
        super(HFG_S2EN,self).__init__()
        pan_dim = 1
        ms_dim = 8
        pan_ms_dim = pan_dim + ms_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck_channels = in_channels * 2
        self.num_heads = num_heads
        self.win_size = win_size

        self.D2_down = Down(in_channels)
        self.spa_down = Down(in_channels)
        self.spe_down = Down(in_channels)
        self.spa_up = Up(self.bottleneck_channels)
        self.spe_up = Up(self.bottleneck_channels)

        self.s2_1 = SSAIB(self.in_channels)
        self.s2_2 = SSAIB(self.bottleneck_channels)
        self.s2_3 = SSAIB(self.in_channels)

        self.point_conv = nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0)
        self.spa_conv1 = SpanConv(in_channels,out_channels,3)
        self.spa_conv2 = SpanConv(self.bottleneck_channels,self.bottleneck_channels,3)
        self.spa_conv3 = SpanConv(out_channels,out_channels,3)

        self.spe_conv1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,5,padding=2,groups=out_channels,bias=False),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels,out_channels,1,1,0),
        )
        self.spe_conv2 = nn.Sequential(
            nn.Conv2d(self.bottleneck_channels,self.bottleneck_channels,3,padding=1,groups=out_channels,bias=False),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.bottleneck_channels,self.bottleneck_channels,1,1,0),
        )
        self.spe_conv3 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,padding=0,groups=out_channels,bias=False),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels,out_channels,1,1,0),
        )
        self.D1 = nn.Sequential(
            nn.Conv2d(pan_ms_dim,out_channels,3,1,1),
            nn.Conv2d(out_channels,out_channels,3,1,1,groups=out_channels, bias=False),
        )

        self.D2 = nn.Sequential(
            nn.Conv2d(pan_ms_dim, out_channels, 1, 1, 0),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1,groups=out_channels, bias=False),
        )

        self.D3 = nn.Sequential(
            nn.Conv2d(pan_ms_dim,out_channels,1,1,0),
            nn.Conv2d(out_channels,out_channels,5,1,2,groups=out_channels,bias=False),
        )
        self.hf_detail_pan = HF_PAN(3)
        self.hf_detail_ms = HF_MS(ms_dim,3)
        self.hf_guided_encoder = HFG_LGEB(self.in_channels, self.num_heads, self.win_size, ffn_expansion_factor,
                                             bias, LayerNorm_type)
        self.hf_bottleneck = HFG_CEB(self.bottleneck_channels, self.num_heads*2, ffn_expansion_factor,
                                                       bias, LayerNorm_type)
        self.hf_guided_decoder = HFG_LGEB(self.in_channels, self.num_heads, self.win_size, ffn_expansion_factor,
                                             bias, LayerNorm_type)

    def forward(self, spa, spe, pan, ms_up):

        _, hf_detail_list_pan = self.hf_detail_pan(pan)
        _, hf_detail_list_ms = self.hf_detail_ms(ms_up)
        D1, D2, D3 = self.D1(torch.cat([hf_detail_list_pan[0],hf_detail_list_ms[0]],dim=1)), self.D2_down(self.D2(torch.cat([hf_detail_list_pan[1],hf_detail_list_ms[1]],dim=1))), self.D3(torch.cat([hf_detail_list_pan[2],hf_detail_list_ms[2]],dim=1))

        ### Stage 1
        spa1 = self.spa_conv1(spa)
        spe1 = self.spe_conv1(spe)
        s2_1 = self.s2_1(spa1, spe1)
        spa1 = self.hf_guided_encoder(s2_1, D1)
        spa1_copy = spa1

        spa1 = self.spa_down(spa1)
        spe1 = self.spe_down(spe1) ### B C*2 H//2 W//2

        ### Stage 2   B C*2 H//2 W//2
        spa2 = self.spa_conv2(spa1)
        spe2 = self.spe_conv2(spe1)
        s2_2 = self.s2_2(spa2, spe2)
        spa2 = self.hf_bottleneck(s2_2, D2)
        spa2 = self.spa_up(spa2)
        spe2 = self.spe_up(spe2)

        ### Stage 3
        spa3 = self.spa_conv3(spa2) + spa
        spe3 = self.spe_conv3(spe2) + spe
        s2_3 = self.s2_3(spa3, spe3)
        spa3 = self.hf_guided_decoder(self.point_conv(torch.cat((s2_3,spa1_copy),dim=1)),D3)

        return spa3




