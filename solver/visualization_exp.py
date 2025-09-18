from solver.basesolver import BaseSolver
import os, torch, time, cv2, importlib
import torch.backends.cudnn as cudnn
from data.data import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from PIL import Image
from data.load_test_data import load_h5py_with_hp, load_h5py
import scipy.io as sio
import math

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def save_feature_map(feature_map, save_dir, filename):
    """将特征图张量保存为可视化图像。"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # 沿着通道维度取平均值，将其降为2D图
    # [B, C, H, W] -> [B, 1, H, W]
    feature_map_2d = torch.mean(feature_map, dim=1, keepdim=True)

    # 因为我们一次只处理一个样本，所以可以压缩掉批次维度
    # [1, 1, H, W] -> [H, W]
    if feature_map_2d.size(0) == 1:
        feature_map_2d = feature_map_2d.squeeze(0).squeeze(0)
    else:
        print(f"警告：特征图的批大小为 {feature_map_2d.size(0)}。仅可视化第一个。")
        feature_map_2d = feature_map_2d[0].squeeze(0)

    # 转换为numpy数组并归一化到 0-255 范围
    img_np = feature_map_2d.cpu().detach().numpy()
    img_normalized = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 应用伪彩色映射（colormap），使特征差异更明显
    img_colored = cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)

    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, img_colored)
    print(f"已保存特征图至: {save_path}")


def visualize_feature_maps_advanced(feature_map, save_dir, filename_prefix, method='grid'):
    """
    将特征图张量保存为可视化图像

    Args:
        feature_map (torch.Tensor): 特征图张量，形状为 [B, C, H, W]。
        save_dir (str): 保存图像的目录。
        filename_prefix (str): 保存文件的前缀。
        method (str): 可视化方法，可选 'mean', 'max', 'grid'。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if feature_map.size(0) > 1:
        print(f"警告: 特征图的批大小为 {feature_map.size(0)}。仅可视化第一个。")
    fm = feature_map[0].detach().cpu()  # Shape: [C, H, W]

    save_path = os.path.join(save_dir, f"{filename_prefix}_{method}.png")

    # --- 方法一：平均值投影
    if method == 'mean':
        aggregated_map = torch.mean(fm, dim=0).numpy()  # Shape: [H, W]
        if aggregated_map.max() == aggregated_map.min():
            img_normalized = np.zeros_like(aggregated_map, dtype=np.uint8)
        else:
            img_normalized = cv2.normalize(aggregated_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_colored = cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)
        cv2.imwrite(save_path, img_colored)

    # --- 方法二：最大值投影
    elif method == 'max':
        aggregated_map = torch.max(fm, dim=0)[0].numpy()  # Shape: [H, W]
        if aggregated_map.max() == aggregated_map.min():
            img_normalized = np.zeros_like(aggregated_map, dtype=np.uint8)
        else:
            img_normalized = cv2.normalize(aggregated_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_colored = cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)
        cv2.imwrite(save_path, img_colored)

    # --- 方法三：通道网格
    elif method == 'grid':
        num_channels = fm.size(0)
        h, w = fm.size(1), fm.size(2)

        # 自动计算网格尺寸
        cols = math.isqrt(num_channels)
        rows = math.ceil(num_channels / cols)

        grid_img = np.zeros((rows * h, cols * w), dtype=np.uint8)

        for i in range(num_channels):
            channel_img = fm[i].numpy()

            # 对每个通道独立进行归一化
            if channel_img.max() > channel_img.min():
                channel_img = cv2.normalize(channel_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                channel_img.fill(0)  # 如果通道全为0，则保持黑色

            # 计算在网格中的位置
            row_idx = i // cols
            col_idx = i % cols
            grid_img[row_idx * h: (row_idx + 1) * h, col_idx * w: (col_idx + 1) * w] = channel_img

        # 应用颜色映射到整个网格
        grid_colored = cv2.applyColorMap(grid_img, cv2.COLORMAP_JET)
        cv2.imwrite(save_path, grid_colored)

    else:
        raise ValueError("未知方法，请选择 'mean', 'max', 或 'grid'")

    print(f"已保存特征图至: {save_path}")
class ExpSolver(BaseSolver):
    def __init__(self, cfg):
        super(ExpSolver, self).__init__(cfg)

        net_name = self.cfg['algorithm'].lower()
        lib = importlib.import_module('model.' + net_name)
        net = lib.Net

        self.model = net(
            dim=self.cfg['data']['n_colors'],
            # base_filter=32,
            # args = self.cfg
        )

    def check(self):
        self.cuda = self.cfg['gpu_mode']
        torch.manual_seed(self.cfg['seed'])
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        if self.cuda:
            torch.cuda.manual_seed(self.cfg['seed'])
            # cudnn.benchmark = False

            gups_list = self.cfg['gpus']
            self.gpu_ids = []
            for str_id in gups_list:
                gid = int(str_id)
                if gid >= 0:
                    self.gpu_ids.append(gid)
            torch.cuda.set_device(self.gpu_ids[0])

            self.model_path = os.path.join(self.cfg['checkpoint'], self.cfg['test']['model'])

            self.model = self.model.cuda(self.gpu_ids[0])
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
            self.model.load_state_dict(torch.load(self.model_path, map_location=lambda storage, loc: storage)['net'])

    def test(self):
        self.model.eval()
        ms, _, pan = load_h5py(self.cfg['test']['data_dir'])

        k = 13
        print(f"\n--- 正在处理并可视化样本 {k} ---")

        with torch.no_grad():
            MS = ms[k, :, :, :].cuda().unsqueeze(dim=0).float()
            PAN = pan[k, 0, :, :].cuda().unsqueeze(dim=0).unsqueeze(dim=1).float()


            output,_,_,spa_maps,spe_maps = self.model(MS, PAN)

            # --- 可视化部分 ---
            # vis_base_dir = os.path.join(self.cfg['test']['save_dir'], f'sample_{k}','visualizations_new')
            vis_base_dir = os.path.join(self.cfg['test']['save_dir'], f'sample_{k}', 'visualizations_advanced')
            print("\n正在可视化空间专家的输出...")
            for expert_idx, f_map in spa_maps.items():
                save_dir = os.path.join(vis_base_dir, f'sample_{k}', 'spatial')
                filename = f'expert_{expert_idx}_output.png'
                visualize_feature_maps_advanced(f_map, save_dir, filename,'mean')

            # 可视化光谱专家（Spectral Expert）的输出
            print("\n正在可视化光谱专家的输出...")
            for expert_idx, f_map in spe_maps.items():
                save_dir = os.path.join(vis_base_dir, f'sample_{k}', 'spectral')
                filename = f'expert_{expert_idx}_output.png'
                visualize_feature_maps_advanced(f_map, save_dir, filename,'mean')
            # print("\n正在可视化各个stage的输出...")
            # for stage_idx, stage_feature in enumerate(lf_ms):
            #     current_stage_name = f'lf_ms{stage_idx}'
            #     save_dir = os.path.join(vis_base_dir, 'lf_ms_outputs')
            #     filename_prefix = f'{current_stage_name}_C{stage_feature.shape[1]}_H{stage_feature.shape[2]}'
            #
            #     # --- 调用新的可视化函数 ---
            #     # 1. 生成网格视图 (最推荐，信息最全)
            #     # visualize_feature_maps_advanced(stage_feature, save_dir, filename_prefix, method='grid')
            #
            #     # 2. 生成最大值投影视图 (快速查看激活区域)
            #     visualize_feature_maps_advanced(stage_feature, save_dir, filename_prefix, method='max')
            #
            #     # 3. (可选) 生成你原来的平均值视图，用于对比
            #     visualize_feature_maps_advanced(stage_feature, save_dir, filename_prefix, method='mean')
            # print("\n正在可视化m_f的输出...")
            # for m_f_idx, m_f in enumerate(m_f_list):
            #     current_mf_name = f'm_f{m_f_idx}'
            #     save_dir = os.path.join(vis_base_dir,'m_f')
            #     filename = f'{current_mf_name}_output.png'
            #     save_feature_map(m_f, save_dir, filename)
            #
            #
            # print("\n正在可视化p_f的输出...")
            # for p_f_idx, p_f in enumerate(p_f_list):
            #     current_pf_name = f'p_f{p_f_idx}'
            #     save_dir = os.path.join(vis_base_dir, 'p_f')
            #     filename = f'{current_pf_name}_output.png'
            #     save_feature_map(p_f, save_dir, filename)


            # print("\n正在可视化stage的输出...")
            # for stage_idx, stage in enumerate(stage_list):
            #     current_stage_name = f'stage{stage_idx}'
            #     save_dir = os.path.join(vis_base_dir,'stage')
            #     filename = f'{current_stage_name}_output.png'
            #     save_feature_map(stage, save_dir, filename)

            # # 可视化空间-1的输出
            # print("\n正在可视化空间的输出...")
            # save_dir = os.path.join(vis_base_dir, f'sample_{k}', 'spatial')
            # filename = f'spa1_output.png'
            # save_feature_map(spa1, save_dir, filename)
            #
            # # 可视化空间-2的输出
            # print("\n正在可视化空间的输出...")
            # save_dir = os.path.join(vis_base_dir, f'sample_{k}', 'spatial')
            # filename = f'spa2_output.png'
            # save_feature_map(spa2, save_dir, filename)
            #
            # # 可视化空间-3的输出
            # print("\n正在可视化空间的输出...")
            # save_dir = os.path.join(vis_base_dir, f'sample_{k}', 'spatial')
            # filename = f'spa3_output.png'
            # save_feature_map(spa3, save_dir, filename)
            #
            # # 可视化光谱的输出
            # print("\n正在可视化光谱的输出...")
            # save_dir = os.path.join(vis_base_dir, f'sample_{k}', 'spectral')
            # filename = f'spe_output.png'
            # save_feature_map(spe, save_dir, filename)

            # # 可视化High-Fre-1的输出
            # print("\n正在可视化High-Fre-1的输出...")
            # save_dir = os.path.join(vis_base_dir, f'sample_{k}', 'High-Fre-1')
            # filename = f'hf1_output.png'
            # save_feature_map(hf1, save_dir, filename)
            #
            # # 可视化High-Fre-2的输出
            # print("\n正在可视化High-Fre-2的输出...")
            # save_dir = os.path.join(vis_base_dir, f'sample_{k}', 'High-Fre-2')
            # filename = f'hf2_output.png'
            # save_feature_map(hf2, save_dir, filename)
            #
            # # 可视化High-Fre-3的输出
            # print("\n正在可视化High-Fre-3的输出...")
            # save_dir = os.path.join(vis_base_dir, f'sample_{k}', 'High-Fre-3')
            # filename = f'hf3_output.png'
            # save_feature_map(hf3, save_dir, filename)

            # 可视化MS-D1的输出
            # print("\n正在可视化lfMS-D1的输出...")
            # save_dir = os.path.join(vis_base_dir, f'sample_{k}', 'lf_MS-D1_max')
            # filename = f'MS_D1_output.png'
            # visualize_feature_maps_advanced(lf_ms[0], save_dir, filename,"max")
            #
            # # 可视化MS-D2的输出
            # print("\n正在可视化lfMS-D2的输出...")
            # save_dir = os.path.join(vis_base_dir, f'sample_{k}', 'lf_MS-D2_max')
            # filename = f'hf2_output.png'
            # visualize_feature_maps_advanced(lf_ms[1], save_dir, filename,"max")
            #
            # # 可视化MS-D3的输出
            # print("\n正在可视化lfMS-D3的输出...")
            # save_dir = os.path.join(vis_base_dir, f'sample_{k}', 'lf_MS-D3_max')
            # filename = f'hf3_output.png'
            # visualize_feature_maps_advanced(lf_ms[2], save_dir, filename,"max")
            #
            # # 可视化PAN-D1的输出
            # print("\n正在可视化lfPAN-D1的输出...")
            # save_dir = os.path.join(vis_base_dir, f'sample_{k}', 'lf_PAN-D1_max')
            # filename = f'PAN_D1_output.png'
            # visualize_feature_maps_advanced(lf_pan[0], save_dir, filename,'max')
            #
            # # 可视化PAN-D2的输出
            # print("\n正在可视化lfPAN-D2的输出...")
            # save_dir = os.path.join(vis_base_dir, f'sample_{k}', 'lf_PAN-D2_max')
            # filename = f'PAN_D2_output.png'
            # visualize_feature_maps_advanced(lf_pan[1], save_dir, filename,'max')
            #
            # # 可视化PAN-D3的输出
            # print("\n正在可视化lfPAN-D3的输出...")
            # save_dir = os.path.join(vis_base_dir, f'sample_{k}', 'lf_PAN-D3_max')
            # filename = f'PAN_D3_output.png'
            # visualize_feature_maps_advanced(lf_pan[2], save_dir, filename,'max')

    def run(self):
        self.check()
        if self.cfg['test']['type'] == 'test':
            # self.dataset = get_test_data(self.cfg, self.cfg['test']['data_dir'])
            # self.data_loader = DataLoader(self.dataset, shuffle=False, batch_size=1,
            #     num_workers=self.cfg['threads'])
            self.test()
        elif self.cfg['test']['type'] == 'eval':
            self.dataset = get_eval_data(self.cfg, self.cfg['test']['data_dir'])
            self.data_loader = DataLoader(self.dataset, shuffle=False, batch_size=1,
                                          num_workers=self.cfg['threads'])
            self.eval()
        else:
            raise ValueError('Mode error!')

