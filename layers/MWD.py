import torch
import pywt
import numpy as np


def modwt_decomposition(data, wavelet='db4', level=3):
    """
    对 [Batch, Channel, Time] 维度的 Tensor 数据进行 MODWT 分解

    参数:
        data: 输入数据 Tensor [Batch, Channel, Time]
        wavelet: wavelet 类型
        level: 分解层数

    返回:
        包含近似系数和细节系数的 Tensor
        格式: [Batch, Channel, Level+1, Time]
    """
    device = data.device
    batch_size, num_channels, time_length = data.shape
    all_coeffs = []

    for i in range(batch_size):
        batch_coeffs = []
        for j in range(num_channels):
            # 将 Tensor 转为 numpy 进行小波变换 (pywt暂不支持直接处理Tensor)
            signal_np = data[i, j, :].cpu().numpy()

            # 进行 MODWT 分解
            coeffs = pywt.mra(signal_np, wavelet=wavelet, level=level, mode='periodization')

            # 将结果转回 Tensor
            coeffs_tensor = torch.stack([torch.from_numpy(c).to(device) for c in coeffs])
            batch_coeffs.append(coeffs_tensor)

        # 堆叠通道结果
        batch_coeffs = torch.stack(batch_coeffs)
        all_coeffs.append(batch_coeffs)

    # 堆叠批次结果 [Batch, Channel, Level+1, Time]
    return torch.stack(all_coeffs)


def modwt_reconstruction(coeffs, wavelet='db4'):
    device = coeffs.device
    batch_size, num_channels, num_levels, time_length = coeffs.shape
    reconstructed = torch.zeros((batch_size, num_channels, time_length), device=device)

    for i in range(batch_size):
        for j in range(num_channels):
            channel_coeffs = coeffs[i, j, :, :]
            # 关键修改：添加 .detach()
            coeff_list = [channel_coeffs[k, :].detach().cpu().numpy() for k in range(num_levels)]
            reconstructed_np = pywt.imra(coeff_list)
            reconstructed[i, j, :] = torch.from_numpy(reconstructed_np).to(device)

    return reconstructed  # 注意缩进修正（原代码return在循环内）

# 测试代码
if __name__ == "__main__":
    # 创建随机Tensor数据 (32批次, 8通道, 1024时间点)
    data = torch.randn(32, 8, 1024).to('cuda')  # 可以使用GPU

    # 进行3层 MODWT 分解
    wavelet = 'db4'
    level = 3
    decomposed = modwt_decomposition(data, wavelet=wavelet, level=level)

    print("分解后形状:", decomposed.shape)  # 应为 [32, 8, 4, 1024]

    # 重构数据
    reconstructed_data = modwt_reconstruction(decomposed)

    print("重构后形状:", reconstructed_data.shape)  # 应为 [32, 8, 1024]

    # 验证重构误差
    error = torch.mean(torch.abs(data - reconstructed_data))
    print(f"重构平均绝对误差: {error.item():.2e}")  # 应该是一个非常小的值