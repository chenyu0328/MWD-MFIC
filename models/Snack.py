import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DynamicSnakeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1):
        super(DynamicSnakeConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

        # 用于生成偏移量的网络
        self.offset_net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 2 * (kernel_size - 1), kernel_size=3, padding=1)
        )

        # 标准卷积权重
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x shape: [B, C, L]
        batch_size, channels, length = x.size()

        # 生成偏移量 [B, 2*(kernel_size-1), L]
        offsets = self.offset_net(x)

        # 重塑偏移量为 [B, kernel_size-1, 2, L]
        offsets = offsets.view(batch_size, self.kernel_size - 1, 2, length)

        # 构建采样网格
        grid = self._get_sampling_grid(offsets)  # [B, kernel_size, L]

        # 应用双线性插值获取采样值
        sampled_features = self._bilinear_sampling(x, grid)  # [B, C, kernel_size, L]

        # 应用卷积权重
        output = self._apply_conv_weights(sampled_features)  # [B, out_channels, L]

        return output

    def _get_sampling_grid(self, offsets):
        """生成采样网格"""
        batch_size, _, _, length = offsets.size()

        # 基础网格坐标
        grid = torch.zeros(batch_size, self.kernel_size, length, device=offsets.device)

        # 中心位置
        grid[:, 0, :] = torch.arange(length, device=offsets.device).float().unsqueeze(0)

        # 累加偏移量生成蛇形路径
        for i in range(1, self.kernel_size):
            # 从通道维度和时间维度获取偏移量
            channel_offset = offsets[:, i - 1, 0, :]
            time_offset = offsets[:, i - 1, 1, :]

            # 计算相对于前一个点的位置
            prev_pos = grid[:, i - 1, :]
            time_step = torch.sign(time_offset) * torch.ceil(torch.abs(time_offset))

            # 防止超出边界
            new_time = prev_pos + time_step
            new_time = torch.clamp(new_time, 0, grid.size(2) - 1)

            grid[:, i, :] = new_time

        return grid

    def _bilinear_sampling(self, x, grid):
        """双线性插值采样特征"""
        batch_size, channels, length = x.size()
        kernel_size = grid.size(1)

        # 扩展输入维度以便于采样
        x = F.pad(x, (self.padding, self.padding), mode='replicate')

        # 初始化采样结果
        sampled = torch.zeros(batch_size, channels, kernel_size, length, device=x.device)

        # 对每个位置进行双线性插值
        for i in range(kernel_size):
            # 计算四个最近的整数坐标
            pos = grid[:, i, :]
            pos0 = torch.floor(pos).long()
            pos1 = torch.clamp(pos0 + 1, 0, x.size(2) - 1)

            # 计算权重
            weight0 = pos1.float() - pos
            weight1 = pos - pos0.float()

            # 采样并加权
            f0 = torch.gather(x, 2, pos0.unsqueeze(1).expand(-1, channels, -1))
            f1 = torch.gather(x, 2, pos1.unsqueeze(1).expand(-1, channels, -1))

            sampled[:, :, i, :] = weight0.unsqueeze(1) * f0 + weight1.unsqueeze(1) * f1

        return sampled

    def _apply_conv_weights(self, sampled_features):
        """应用卷积权重到采样的特征上"""
        batch_size, channels, kernel_size, length = sampled_features.size()

        # 重塑为标准卷积形式
        sampled_features = sampled_features.view(batch_size, channels * kernel_size, length)

        # 重塑权重以应用分组卷积
        weight = self.weight.view(self.out_channels, channels * kernel_size, 1)

        # 应用卷积
        output = F.conv1d(sampled_features, weight, self.bias, stride=self.stride,
                          padding=0, dilation=1, groups=1)

        return output


class TimeSeriesPredictor(nn.Module):
    def __init__(self, in_channels, seq_len, pred_len, hidden_channels=64):
        super(TimeSeriesPredictor, self).__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 特征提取层
        self.feature_extractor = nn.Sequential(
            DynamicSnakeConv(in_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            DynamicSnakeConv(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU()
        )

        # 预测层
        self.predictor = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, in_channels, kernel_size=3, padding=1)
        )

        # 用于将输入序列映射到预测序列长度的层
        self.length_adjust = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x shape: [B, C, L_in]
        batch_size, channels, length = x.size()

        # 特征提取
        features = self.feature_extractor(x)

        # 调整序列长度
        features = self.length_adjust(features)  # [B, hidden_channels, L_out]

        # 预测
        output = self.predictor(features)

        return output


# 简单示例
# def test_dynamic_snake_conv():
#     # 设置随机种子以确保结果可复现
#     torch.manual_seed(42)
#
#     # 创建模型
#     in_channels = 5  # 输入通道数
#     seq_len = 10  # 输入序列长度
#     pred_len = 5  # 预测序列长度
#     model = TimeSeriesPredictor(in_channels, seq_len, pred_len)
#
#     # 创建测试输入
#     batch_size = 2
#     x = torch.randn(batch_size, in_channels, seq_len)
#
#     # 前向传播
#     with torch.no_grad():
#         output = model(x)
#
#     print(f"输入形状: {x.shape} ([Batch, Channel, Input length])")
#     print(f"输出形状: {output.shape} ([Batch, Channel, Output length])")
#     print("输出示例:")
#     print(output[0, :, :5])  # 打印第一个样本的前5个时间步
#
#
# if __name__ == "__main__":
#     test_dynamic_snake_conv()