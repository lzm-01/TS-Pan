import torch
import importlib
import time
from thop import profile
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# --- 1. 你的原始模型设置 ---
model_name = 'your_model_name'
net_name = model_name.lower()
lib = importlib.import_module('model.' + net_name)
net = lib.Net

model = net(
    # dim = 32
)

# --- 2. 准备输入数据和设备 ---
# 使用和FLOPs计算时相同的输入尺寸
input_data = torch.randn(1, 8, 64, 64)
input_data1 = torch.randn(1, 1, 256, 256)

# 确定运行设备 (优先使用GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 将模型和输入数据移动到指定设备
model.to(device)
input_data = input_data.to(device)
input_data1 = input_data1.to(device)

# --- 3. 计算 FLOPs 和 Params (使用 thop) ---
# 确保模型在评估模式
model.eval()
# thop需要在CPU上运行模型和输入
flops, params = profile(model, inputs=(input_data, input_data1), verbose=False)
print("-" * 30)
print("Model Complexity Analysis")
print(f"FLOPs: {flops / 1e9:.4f} GFLOPs")
print(f"Params: {params / 1e6:.4f} M")
print("-" * 30)

# --- 4. 测量推理速度 ---
print("\n" + "-" * 30)
print("Inference Speed Analysis")

# 如果之前移到了CPU,将模型移回GPU
model.to(device)

# 设置为评估模式并禁用梯度计算
model.eval()
with torch.no_grad():
    # 预热运行 (Warm-up)
    print("Warming up...")
    warmup_runs = 20
    for _ in range(warmup_runs):
        _ = model(input_data, input_data1)

    # 精确计时
    print("Measuring inference time...")
    num_runs = 100
    start_time = time.time()

    # 对于GPU计时，需要使用事件或同步
    if device.type == 'cuda':
        # 清空缓存并同步，确保计时准确
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_runs):
            _ = model(input_data, input_data1)
        end_event.record()
        torch.cuda.synchronize()  # 等待所有GPU任务完成

        # 计算总时间（毫秒）
        elapsed_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = elapsed_time_ms / num_runs
    else:  # CPU 计时
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = model(input_data, input_data1)
        end_time = time.perf_counter()

        # 计算总时间（毫秒）
        elapsed_time_ms = (end_time - start_time) * 1000
        avg_time_ms = elapsed_time_ms / num_runs

    fps = 1000 / avg_time_ms

    print(f"Performed {num_runs} runs.")
    print(f"Average inference time: {avg_time_ms:.4f} ms/frame")
    print(f"Inference speed (FPS): {fps:.2f} FPS")
print("-" * 30)

# --- 5. 测量峰值显存占用 (仅限GPU) ---
if device.type == 'cuda':
    print("\n" + "-" * 30)
    print("Peak Memory Usage Analysis (GPU only)")

    # 清空CUDA缓存并重置峰值统计
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # 再次运行一次模型以记录内存使用
    with torch.no_grad():
        _ = model(input_data, input_data1)

    # 获取峰值显存占用（单位：字节）
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)

    print(f"Peak GPU memory allocated: {peak_memory_mb:.2f} MB")
    print("-" * 30)
else:
    print("\nMemory usage analysis is only available for CUDA devices.")
