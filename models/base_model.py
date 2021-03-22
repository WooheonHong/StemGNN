import math

from utils.graph_utils import calculate_scaled_laplacian
from utils.graph_utils import calculate_normalized_laplacian
from utils.graph_utils import calculate_randomwalk_normalized_laplacian

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 1d conv + Gated Linear Unit = gated CNN
class GCNN(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3):
        super(GCNN, self).__init__()
        self.filter_conv = nn.Conv2d(
            input_channel, output_channel, kernel_size=(1, kernel_size),
        )
        self.gate_conv = nn.Conv1d(
            input_channel, output_channel, kernel_size=(1, kernel_size),
        )

    def forward(self, x):
        return torch.mul(self.filter_conv(x), torch.sigmoid(self.gate_conv(x)))


class StemGNN_Block(nn.Module):
    def __init__(
        self,
        time_step,
        unit,
        kernel_size,
        gcnn_channel,
        gconv_channel,
        multi_channel,
        dropout_rate,
        stack_cnt=0,
    ):
        super(StemGNN_Block, self).__init__()
        self.time_step = time_step
        self.unit = unit
        self.stack_cnt = stack_cnt

        self.residual_channel = gconv_channel
        self.kernel_size = kernel_size
        self.gcnn_channel = gcnn_channel
        self.gconv_channel = gconv_channel
        self.multi_channel = multi_channel

        self.batch_norm = nn.BatchNorm2d(self.multi_channel)
        self.relu = nn.ReLU()

        self.residual_conv = nn.ModuleList()
        self.gcnn_real = nn.ModuleList()
        self.gcnn_imag = nn.ModuleList()
        # block 1
        self.residual_conv.append(nn.Conv2d(1, self.residual_channel, kernel_size=(1, 1)))
        self.gcnn_real.append(GCNN(1, self.gcnn_channel, self.kernel_size))
        self.gcnn_imag.append(GCNN(1, self.gcnn_channel, self.kernel_size))

        # block 2
        self.residual_conv.append(
            nn.Conv2d(self.gconv_channel, self.residual_channel, kernel_size=(1, 1))
        )
        self.gcnn_real.append(GCNN(self.gconv_channel, self.gcnn_channel, self.kernel_size))
        self.gcnn_imag.append(GCNN(self.gconv_channel, self.gcnn_channel, self.kernel_size))

        self.weight = nn.Parameter(
            torch.Tensor(self.gcnn_channel, self.gconv_channel)
        )  # [K+1, 1, in_c, out_c]
        nn.init.kaiming_normal_(self.weight)

        self.fc = nn.Sequential(
            nn.Conv2d(self.gconv_channel, self.gconv_channel, kernel_size=(1, 1),),
            # nn.Sigmoid(),
            # nn.Softmax(dim=1),
        )
        self.forecast = nn.Sequential(
            nn.Conv2d(self.gconv_channel, self.multi_channel, kernel_size=(1, 1),),
            nn.ReLU(),
            nn.BatchNorm2d(self.multi_channel),
            nn.Dropout(p=dropout_rate),
        )
        self.backcast = nn.Sequential(
            nn.Conv2d(self.gconv_channel, self.multi_channel, kernel_size=(1, 1),),
            nn.ReLU(),
            nn.BatchNorm2d(self.multi_channel),
            nn.Dropout(p=dropout_rate),
        )

    def spe_seq_cell(self, x):
        # for real input, negative frequency redundant
        ffted = torch.fft.rfft(x)  # 12 -> (12/6) + 1
        # ffted = torch.stft(x, 4)  # 12 -> (12/6) + 1
        # print("after fft:", ffted.shape)
        real = self.gcnn_real[self.stack_cnt](ffted.real)  # (12/6) + 1 -> (12/6) + 1 - 2
        imag = self.gcnn_imag[self.stack_cnt](ffted.imag)
        time_step_as_inner = torch.stack((real, imag), -1)
        # print("time_step_as_inner: ", time_step_as_inner.shape)
        time_step_as_inner = torch.view_as_complex(time_step_as_inner)
        iffted = torch.fft.irfft(time_step_as_inner)  # (((12/6) + 1 - 2 ) - 1) * 2-> 8
        # iffted = torch.istft(time_step_as_inner, 4)
        # print("iffted shape: ", iffted.shape)

        return iffted

    def forward(self, input, Lambda_, U):
        GFT = torch.transpose(U, 0, 1)
        gconv_operator = torch.diag_embed(Lambda_)  # make diagonal matrix
        IGFT = U

        x_residual = self.relu(self.residual_conv[self.stack_cnt](input))
        # print("x residual shape: ", x_residual.shape)
        x = torch.einsum("binw, nn -> binw", input, GFT)
        x = self.spe_seq_cell(x)
        x = torch.einsum("binw, nn -> binw", x, gconv_operator)  # gconv
        x = torch.einsum("binw, io -> bonw", x, self.weight)
        x = torch.einsum("bonw, nn -> bonw", x, IGFT)

        x = self.fc(x)

        x = self.relu(
            x_residual[:, :, :, (self.kernel_size - 1) * 2 : x_residual.shape[3]] + x
        )

        if self.stack_cnt == 0:
            backcast_source = x_residual[
                :, :, :, (self.kernel_size - 1) * 2 : x_residual.shape[3]
            ] - self.relu(x)

        else:
            backcast_source = x

        forecast = self.forecast(x)
        backcast = self.backcast(backcast_source)

        return forecast, backcast, backcast_source


class OutputLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(OutputLayer, self).__init__()
        self.glu = nn.GLU(dim=1)
        self.batch_norm = nn.BatchNorm2d(int(c_in / 2))
        self.conv2d = nn.Conv2d(int(c_in / 2), int(c_in / 2), kernel_size=(1, 1),)
        self.sigmoid = nn.Sigmoid()
        self.out_conv = nn.Conv2d(int(c_in / 2), c_out, kernel_size=(1, 1),)

    def forward(self, x):
        x = self.glu(x)
        x = self.batch_norm(x)
        x = self.conv2d(x)
        x = self.sigmoid(x)
        x = self.out_conv(x)
        return x


class Model(nn.Module):
    def __init__(
        self,
        units,
        stack_cnt,
        time_step,
        attention_channel,
        is_randomwalk_laplacian,
        kernel_size,
        gcnn_channel,
        gconv_channel,
        multi_channel,
        horizon,
        dropout_rate=0.5,
        leaky_rate=0.2,
        device="cuda",
    ):
        super(Model, self).__init__()
        self.unit = units  # # of nodes
        self.stack_cnt = stack_cnt  # 2
        self.alpha = leaky_rate
        self.time_step = time_step
        self.horizon = horizon
        self.attention_channel = attention_channel
        self.kernel_size = kernel_size
        self.gcnn_channel = gcnn_channel
        self.gconv_channel = gconv_channel

        self.GRU = nn.GRU(self.unit, self.unit, batch_first=True, num_layers=1,)
        self.weight_key = nn.Parameter(torch.zeros(size=(1, self.attention_channel)))
        nn.init.kaiming_uniform_(self.weight_key.data)
        self.weight_query = nn.Parameter(torch.zeros(size=(1, self.attention_channel)))
        nn.init.kaiming_uniform_(self.weight_query.data)
        self.weight_attention = nn.Parameter(
            torch.zeros(size=(self.attention_channel * self.unit, self.unit))
        )
        nn.init.kaiming_normal_(self.weight_attention.data)

        self.multi_channel = multi_channel
        self.is_randomwalk_laplacian = is_randomwalk_laplacian

        self.stemgnn_block = nn.ModuleList()
        self.stemgnn_block.extend(
            [
                StemGNN_Block(
                    self.time_step,
                    self.unit,
                    self.kernel_size,
                    self.gcnn_channel,
                    self.gconv_channel,
                    self.multi_channel,
                    dropout_rate,
                    stack_cnt=i,
                )
                for i in range(self.stack_cnt)
            ]
        )
        self.forecast_output = OutputLayer(self.multi_channel, 1)
        self.backcast_output = OutputLayer(self.multi_channel, self.time_step)

        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.device = device
        self.to(self.device)

    def latent_correlation_layer(self, x):
        batch_size = x.shape[0]
        output, hidden = self.GRU(x)
        output = output[:, -1:, :]
        attention = self.self_graph_attention(output)
        attention = torch.mean(attention, dim=0)  # average of batch
        degree = torch.sum(attention, dim=1)
        # laplacian is sym or not
        attention = 0.5 * (attention + attention.T)
        if self.is_randomwalk_laplacian:
            laplacian = calculate_randomwalk_normalized_laplacian(attention)
        else:
            laplacian = calculate_normalized_laplacian(attention).to(self.device)
        Lambda_, U = torch.symeig(laplacian, eigenvectors=True)
        # Lambda_ = self.relu(Lambda_)
        # U = self.relu(U)
        return Lambda_, U, attention

    def self_graph_attention(self, input):
        batch_size = input.shape[0]
        input = input.permute(0, 2, 1).contiguous()
        key = torch.einsum("bni, io -> bno", input, self.weight_key)
        query = torch.einsum("bni, io -> bno", input, self.weight_query)
        data = key.repeat(1, 1, self.unit).view(
            batch_size, self.unit * self.unit, -1
        ) + query.repeat(1, self.unit, 1)
        data = data.squeeze(2)
        data = data.view(batch_size, self.unit, -1)
        data = torch.matmul(data, self.weight_attention)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    def forward(self, x):
        Lambda_, U, attention = self.latent_correlation_layer(x)
        x = x.permute(0, 2, 1).contiguous()
        X = x.unsqueeze(1).contiguous()  # batch, feature, node, window
        result_forecast = []
        result_backcast = []
        for stack_i in range(self.stack_cnt):
            forecast, backcast, X = self.stemgnn_block[stack_i](X, Lambda_, U)
            result_forecast.append(forecast)
            result_backcast.append(backcast)
        # print("result_forecast shape 0: ", result_forecast[0].shape)
        # print("result_forecast shape 1: ", result_forecast[1].shape)
        forecast = (
            result_forecast[0][:, :, :, (3 - 1) * 2 : result_forecast[0].shape[3]]
            + result_forecast[1]
        )
        if len(result_backcast) > 1:
            backcast = (
                result_backcast[0][:, :, :, (3 - 1) * 2 : result_backcast[0].shape[3]]
                + result_backcast[1]
            )

        else:
            backcast = result_backcast[0][
                :, :, :, (3 - 1) * 2 : result_backcast[0].shape[3]
            ]

        forecast = self.forecast_output(forecast)[
            :, :, :, forecast.shape[3] - 1 : forecast.shape[3]
        ]
        forecast = forecast.squeeze(3)
        # forecast = self.forecast_output(forecast)
        # forecast = torch.mean(forecast, dim=3)

        backcast = self.backcast_output(backcast)[
            :, :, :, backcast.shape[3] - 1 : backcast.shape[3]
        ]
        backcast = backcast.squeeze(3)
        # backcast = self.backcast_output(backcast)
        # backcast = torch.mean(backcast, dim=3)

        if forecast.size()[-1] == 1:
            return forecast.unsqueeze(1).squeeze(-1), backcast, attention
            # return forecast.unsqueeze(1).squeeze(-1), attention
        else:
            return forecast, backcast, attention
            # return forecast, attention
