import torch
import torch.nn as nn
import torch.nn.functional as F


def get_b3_spline_filter_2d():  
    kernel_1d = torch.tensor([1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16], dtype=torch.float32)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.unsqueeze(0).unsqueeze(0)


class HF_PAN(nn.Module):
    """针对PAN高频提取"""
    def __init__(self, num_levels, kernel_size=5, init_method='b3_spline', padding_mode='reflect'):
        super().__init__()
        self.num_levels = num_levels
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size

        self.h_filters = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, kernel_size, kernel_size))
            for _ in range(num_levels)
        ])
        self._initialize_filters(init_method)

    def _initialize_filters(self, method='b3_spline'):
        """初始化所有滤波器权重"""
        with torch.no_grad():
            for level, h_filter in enumerate(self.h_filters):
                if method == 'b3_spline' and self.kernel_size == 5:
                    init_filter = get_b3_spline_filter_2d()
                    h_filter.copy_(init_filter)
                elif method == 'gaussian':
                    # 每个级别使用不同的sigma
                    sigma = (self.kernel_size / 6.0) * (1 + 0.2 * level)
                    k = self.kernel_size // 2
                    x = torch.arange(-k, k + 1, dtype=torch.float32)
                    gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
                    gauss_1d = gauss_1d / gauss_1d.sum()
                    gauss_2d = torch.outer(gauss_1d, gauss_1d)
                    h_filter.copy_(gauss_2d.unsqueeze(0).unsqueeze(0))
                else:
                    nn.init.xavier_uniform_(h_filter)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            if x.shape[0] > 1 and x.shape[0] <= 4:
                x = x.unsqueeze(0)
            else:
                x = x.unsqueeze(1)

        B, C, H_in, W_in = x.shape
        detail_coeffs_list = []
        lf_list = []
        current_approx = x

        for j in range(self.num_levels):
            dilation_factor = 2 ** j
            # dilation_factor = j + 1
            current_approx_reshaped = current_approx.reshape(B * C, 1, H_in, W_in)

            pad_h_eff = (self.kernel_size - 1) // 2 * dilation_factor
            pad_w_eff = (self.kernel_size - 1) // 2 * dilation_factor

            padded_input = F.pad(current_approx_reshaped,
                                 (pad_w_eff, pad_w_eff, pad_h_eff, pad_h_eff),
                                 mode=self.padding_mode)

            low_pass_output_reshaped = F.conv2d(
                padded_input, self.h_filters[j], stride=1,
                dilation=dilation_factor, groups=1)

            next_approx = low_pass_output_reshaped.reshape(B, C, H_in, W_in)
            lf_list.append(next_approx)
            detail_coeffs = current_approx - next_approx
            detail_coeffs_list.append(detail_coeffs)
            current_approx = next_approx

        return lf_list, detail_coeffs_list


class HF_MS(nn.Module):
    """针对MS高频提取"""

    def __init__(self, num_channels, num_levels, kernel_size=5, init_method='b3_spline', padding_mode='reflect'):
        super().__init__()
        self.num_channels = num_channels
        self.num_levels = num_levels
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size

        self.h_filters_modulelist = nn.ModuleList() 
        for _ in range(num_channels):
            level_filters = nn.ParameterList() 
            for _ in range(num_levels):
                level_filters.append(nn.Parameter(torch.zeros(1, 1, kernel_size, kernel_size)))
            self.h_filters_modulelist.append(level_filters)

        self._initialize_filters(init_method)

    def _initialize_filters(self, method='b3_spline'):
        """初始化所有滤波器权重"""
        with torch.no_grad():
            for c in range(self.num_channels):
                for level in range(self.num_levels):
                    h_filter_param = self.h_filters_modulelist[c][level]
                    if method == 'b3_spline' and self.kernel_size == 5:
                        init_filter = get_b3_spline_filter_2d()
                        h_filter_param.copy_(init_filter)
                    elif method == 'gaussian':
                        sigma = (self.kernel_size / 6.0) * (1 + 0.2 * level) 
                        k = self.kernel_size // 2
                        x_coords = torch.arange(-k, k + 1, dtype=torch.float32)
                        gauss_1d = torch.exp(-0.5 * (x_coords / sigma) ** 2)
                        gauss_1d = gauss_1d / gauss_1d.sum()
                        gauss_2d = torch.outer(gauss_1d, gauss_1d)
                        h_filter_param.copy_(gauss_2d.unsqueeze(0).unsqueeze(0))
                    else:
                        nn.init.xavier_uniform_(h_filter_param)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            if x.shape[0] > 1 and x.shape[0] <= 4:  
                x = x.unsqueeze(0)
            else:
                x = x.unsqueeze(1)

        B, C, H_in, W_in = x.shape

        if C != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} input channels based on module init, got {C}")

        detail_coeffs_list = []
        approx_coeffs_list = []  # To store A0, A1, ..., AN
        current_approx = x
        approx_coeffs_list.append(current_approx.clone())

        for j in range(self.num_levels):
            dilation_factor = 2 ** j
            # dilation_factor = j+1
            pad_h_eff = (self.kernel_size - 1) // 2 * dilation_factor
            pad_w_eff = (self.kernel_size - 1) // 2 * dilation_factor

            channel_outputs = []
            for c in range(C):  # Iterate through actual input channels
                channel_data = current_approx[:, c:c + 1, :, :]

                padded_input = F.pad(channel_data,
                                     (pad_w_eff, pad_w_eff, pad_h_eff, pad_h_eff),
                                     mode=self.padding_mode)

                if c < len(self.h_filters_modulelist) and j < len(self.h_filters_modulelist[c]):
                    current_filter = self.h_filters_modulelist[c][j]
                else:
                    current_filter = self.h_filters_modulelist[c][j]

                low_pass_output = F.conv2d(
                    padded_input, current_filter, stride=1,
                    dilation=dilation_factor, groups=1)

                channel_outputs.append(low_pass_output)

            next_approx = torch.cat(channel_outputs, dim=1)
            detail_coeffs = current_approx - next_approx
            detail_coeffs_list.append(detail_coeffs)
            approx_coeffs_list.append(next_approx.clone())
            current_approx = next_approx

        return approx_coeffs_list, detail_coeffs_list


