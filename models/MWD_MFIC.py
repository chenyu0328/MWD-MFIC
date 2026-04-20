__all__ = ['MWD_MFIC']

import pywt
# Cell
import torch
from torch import nn
from yacs.config import CfgNode as CN
from layers.RevIN import RevIN
from pytorch_wavelets import DWT1D, IDWT1D
from layers.PatchTST_layers import Transpose
from layers.MWD import modwt_decomposition, modwt_reconstruction

class Model(nn.Module):
    def __init__(self, configs, act: str = "gelu", **kwargs):
        super().__init__()
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        individual = configs.individual
        self.learning_rate = configs.learning_rate
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        self.batch_size = configs.batch_size

        # model
        head_dropout = configs.head_dropout
        wavelet_layers = configs.wavelet_layers
        wavelet_type = configs.wavelet_type
        wavelet_mode = configs.wavelet_mode
        wavelet_dim = configs.wavelet_dim
        hidden_wavelet_dim = configs.hidden_wavelet_dim

        cfg = CN()

        configs = vars(configs)

        for k in configs.keys():
            cfg[k] = configs[k]
        self.model = MTST_TCN_backbone(c_in=c_in, context_window=context_window, target_window=target_window, individual=individual, revin=revin,
                                   affine=affine, subtract_last=subtract_last, cfg=cfg,act=act,wavelet_layers=wavelet_layers,wavelet_type=wavelet_type,
                                    wavelet_mode=wavelet_mode,wavelet_dim=wavelet_dim,hidden_wavelet_dim=hidden_wavelet_dim, head_dropout=head_dropout, **kwargs)


    def forward(self, x):
        z = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
        z, draw_list, attn_ls = self.model(z)
        z = z.permute(0, 2, 1)
        return z, draw_list, attn_ls

class MTST_TCN_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, individual = False, revin = True, affine = True,
                 subtract_last = False, cfg=CN(),act: str = "gelu",wavelet_layers=5, wavelet_type='haar',wavelet_mode='symmetric',
                 wavelet_dim=64,hidden_wavelet_dim=64, head_dropout=0, **kwargs):
        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        self.target_window = target_window
        self.context_window = context_window
        self.individual = individual
        self.channels = c_in
        self.wavelet_layers = wavelet_layers
        self.wavelet_type = wavelet_type
        self.wavelet_mode = wavelet_mode
        self.wavelet_dim = wavelet_dim
        self.hidden_wavelet_dim = hidden_wavelet_dim
        # Wavelet
        self.dwt = DWT1D(J=wavelet_layers, wave=wavelet_type, mode=wavelet_mode)
        self.idwt = IDWT1D(wave=wavelet_type, mode=wavelet_mode)

        temp_seq = torch.rand(1, 1, context_window)

        temp_seq_yl, temp_seq_yh = self.dwt(temp_seq)

        seq_len_J = [y.shape[-1] for y in temp_seq_yh] + [temp_seq_yl.shape[-1]]

        temp_pred = torch.rand(1, 1, context_window)
        temp_pred_yl, temp_pred_yh = self.dwt(temp_pred)
        pred_len_J = [y.shape[-1] for y in temp_pred_yh] + [temp_pred_yl.shape[-1]]

        self.in_proj_h = nn.ModuleList([
            nn.Linear(seq_len_J[i], wavelet_dim)
            for i in range(wavelet_layers)
        ])
        self.in_proj_l = nn.Linear(seq_len_J[-1], wavelet_dim)

        self.out_proj_h = nn.ModuleList([
            nn.Sequential(
                nn.Linear(wavelet_dim, wavelet_dim * 2),
                nn.ReLU() if act == 'relu' else nn.GELU(),
                nn.Linear(wavelet_dim * 2, pred_len_J[i])
            ) for i in range(wavelet_layers)
        ])
        self.out_proj_l = nn.Sequential(
            nn.Linear(wavelet_dim, wavelet_dim * 2),
            nn.ReLU if act == 'relu' else nn.GELU(),
            nn.Linear(wavelet_dim * 2, pred_len_J[-1])
        )

        self.Linear_1 = nn.Linear(context_window, context_window)

        self.Linear_ = nn.Linear(context_window, target_window)

        self.waveletLinearLayers_h = nn.ModuleList()
        for i in range(wavelet_layers+1):
            self.waveletLinearLayers_h.append(nn.Sequential(
                nn.Linear(wavelet_dim, hidden_wavelet_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(hidden_wavelet_dim, wavelet_dim),
                nn.Dropout(head_dropout)
            ))

        self.waveletLinearLayers_l = nn.Sequential(
            nn.Linear(wavelet_dim, hidden_wavelet_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_wavelet_dim, wavelet_dim),
            nn.Dropout(head_dropout)
        )
        self.LayerNorm =  nn.LayerNorm(wavelet_dim)
        self.conv_kernel = 1
        self.conv = nn.Conv1d(in_channels=self.wavelet_layers+1, out_channels=self.wavelet_layers+1, kernel_size=self.conv_kernel, stride=1,
                              padding=(self.conv_kernel - 1) // 2, padding_mode="zeros", bias=False)
        self.w1 = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor(0.8), requires_grad=True)

    def forward(self, x):
        # norm
        if self.revin:
            x = x.permute(0, 2, 1) #  [B, C, L] -> [B, L, C]
            x = self.revin_layer(x, 'norm') #
            x = x.permute(0, 2, 1) #  [B, C, L]

        # yl: [B, M, L/2^J], yh: [[B, M, L/2^j]]
        yl, yh = self.dwt(x)  #

        for i in range(len(yh)):
            yh[i] = self.in_proj_h[i](yh[i]).unsqueeze(-2)
        yl = self.in_proj_l(yl).unsqueeze(-2)

        # [[B, M, 1, D]] -> [B, M, J, D]
        enc_in1 = torch.cat(yh, dim=-2)
        enc_in2 = torch.cat([yl], dim=-2)

        y = torch.cat((enc_in1, enc_in2), dim=-2)
        y1 = self.conv(y.reshape(int(y.size(0) * self.channels), self.wavelet_layers + 1, y.size(-1))).reshape(-1,
                                                                                                                  self.channels,
                                                                                                                  self.wavelet_layers + 1,
                                                                                                                   y.size(
                                                                                                                   -1))
        y = y1 * self.w1 + y * self.w2

        enc_in1 = y[:, :, :self.wavelet_layers, :]
        enc_in2 = y[:, :, self.wavelet_layers:, :]
        o_yh = []
        for i in range(len(yh)):
            input = enc_in1[:, :, i, :]
            output = input + self.waveletLinearLayers_h[i](input)
            output = self.LayerNorm(output)
            o_yh.append(output)
        enc_in1 = torch.stack(o_yh, dim=-2)
        o_yl = []
        input = enc_in2[:, :, 0, :]#shape: [B, C, D]
        output = input + self.waveletLinearLayers_l(input)
        output = self.LayerNorm(output)
        o_yl.append(output)

        enc_in2 = torch.stack(o_yl, dim=-2)

        enc_in = torch.cat((enc_in1, enc_in2), dim=-2)
        enc_in = list(torch.unbind(enc_in, dim=-2))
        for i in range(len(yh)):
            yh[i] = self.out_proj_h[i](enc_in[i])
        yl = self.out_proj_l(enc_in[-1])
        x1 = self.idwt((yl, yh)).permute(0, 2, 1)[:, :, :self.channels].permute(0, 2, 1)
        output1 = self.Linear_(x1)
        z = output1
        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)

            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z, [], []

def detect_and_correct_anomalies(tensor, threshold=3, correction_method='mean'):
    batch_size, channels, input_length = tensor.shape
    rolling_mean = tensor.mean(dim=-1, keepdim=True)  #  [Batch, Channel, 1]
    rolling_std = tensor.std(dim=-1, keepdim=True)  # [Batch, Channel, 1]
    rolling_mean = rolling_mean.expand(-1, -1, input_length)  #  [Batch, Channel, Input Length]
    rolling_std = rolling_std.expand(-1, -1, input_length)  # [Batch, Channel, Input Length]
    z_score = (tensor - rolling_mean) / (rolling_std + 1e-8)
    anomaly_mask = torch.abs(z_score) > threshold
    corrected_tensor = tensor.clone()

    if correction_method == 'mean':
        corrected_tensor[anomaly_mask] = rolling_mean[anomaly_mask]
    elif correction_method == 'median':
        rolling_median = torch.median(tensor, dim=-1, keepdim=True)[0]
        corrected_tensor[anomaly_mask] = rolling_median[anomaly_mask]
    elif correction_method == 'interpolate':# interpolate
        for i in range(tensor.shape[0]):  # Batch
            for j in range(tensor.shape[1]):  # Channel
                # 对每个序列进行线性插值
                x = torch.arange(input_length) #  tensor([0, 1, 2, ..., input_length-1])
                y = tensor[i, j, :]            # shape: (input_length,)
                mask = anomaly_mask[i, j, :]   # shape: (input_length,)
                if mask.any():
                    y[mask] = torch.interp(x[mask], x[~mask], y[~mask])
                corrected_tensor[i, j, :] = y
    return corrected_tensor
