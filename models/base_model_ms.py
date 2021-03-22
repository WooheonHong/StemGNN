from utils.graph_utils import calculate_normalized_laplacian
from utils.graph_utils import calculate_randomwalk_normalized_laplacian
from utils.graph_utils import cheb_polynomial


import torch
import torch.nn as nn
import torch.nn.functional as F

# 1d conv + Gated Linear Unit = gated CNN
class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))


class StockBlockLayer(nn.Module):
    def __init__(self, time_step, unit, multi_layer, dropout_rate, stack_cnt=0):
        super(StockBlockLayer, self).__init__()
        self.time_step = time_step
        self.unit = unit
        self.dropout = nn.Dropout(p=dropout_rate)
        self.stack_cnt = stack_cnt
        self.multi = multi_layer
        self.weight = nn.Parameter(
            torch.Tensor(
                1, 3 + 1, 1, self.time_step * self.multi, self.multi * self.time_step
            )
        )  # [K+1, 1, in_c, out_c]
        # nn.init.xavier_normal_(self.weight)
        nn.init.kaiming_normal_(self.weight)
        self.forecast = nn.Linear(self.time_step * self.multi, self.time_step * self.multi)
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step)
        if self.stack_cnt == 0:
            self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)
        self.backcast_short_cut = nn.Linear(self.time_step, self.time_step)
        self.batch_norm_forecast = nn.BatchNorm1d(self.unit)
        self.relu = nn.ReLU()
        self.GLUs = nn.ModuleList()
        self.output_channel = 4 * self.multi
        for i in range(3):
            if i == 0:
                self.GLUs.append(
                    GLU(self.time_step * 4, self.time_step * self.output_channel)
                )
                self.GLUs.append(
                    GLU(self.time_step * 4, self.time_step * self.output_channel)
                )
            elif i == 1:
                self.GLUs.append(
                    GLU(
                        self.time_step * self.output_channel,
                        self.time_step * self.output_channel,
                    )
                )
                self.GLUs.append(
                    GLU(
                        self.time_step * self.output_channel,
                        self.time_step * self.output_channel,
                    )
                )
            else:
                self.GLUs.append(
                    GLU(
                        self.time_step * self.output_channel,
                        self.time_step * self.output_channel,
                    )
                )
                self.GLUs.append(
                    GLU(
                        self.time_step * self.output_channel,
                        self.time_step * self.output_channel,
                    )
                )

    def spe_seq_cell(self, input):
        batch_size, k, input_channel, node_cnt, time_step = input.size()
        input = input.view(batch_size, -1, node_cnt, time_step)
        ffted = torch.rfft(input, 1, onesided=False)
        real = (
            ffted[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        )
        img = (
            ffted[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        )
        for i in range(3):
            real = self.GLUs[i * 2](real)  # batch, node, time_step * output_channel
            img = self.GLUs[2 * i + 1](img)
        real = real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        time_step_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)
        iffted = torch.irfft(time_step_as_inner, 1, onesided=False)
        return iffted

    def forward(self, x, mul_L):
        mul_L = mul_L.unsqueeze(1)  # 4, 1, node, node
        x = x.unsqueeze(1)  # batch, 1, node, window
        gfted = torch.matmul(mul_L, x)  # batch, 4, 1, node, window
        gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)  # batch 4, 1, node, window
        igfted = torch.matmul(gconv_input, self.weight)  # batch 4, 1, node, window *
        igfted = torch.sum(igfted, dim=1)
        # forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
        forecast_source = self.relu(self.forecast(igfted).squeeze(1))
        # forecast_source = self.batch_norm_forecast(forecast_source)
        forecast = self.forecast_result(forecast_source)
        # forecast = self.batch_norm_forecast(forecast)
        if self.stack_cnt == 0:
            backcast_short = self.backcast_short_cut(x).squeeze(1)
            # backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)
            backcast_source = self.relu(self.backcast(igfted) - backcast_short)
        else:
            backcast_source = None

        return forecast, backcast_source


class Model(nn.Module):
    def __init__(
        self,
        units,
        stack_cnt,
        time_step,
        multi_layer,
        is_randomwalk_laplacian,
        attention_layer,
        horizon=1,
        dropout_rate=0.5,
        leaky_rate=0.2,
        device="cuda",
    ):
        super(Model, self).__init__()
        self.unit = units  # # of nodes
        self.stack_cnt = stack_cnt  # 2
        self.alpha = leaky_rate
        self.time_step = time_step  # window size
        self.horizon = horizon
        self.attention_layer = attention_layer
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, self.attention_layer)))
        # nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        nn.init.kaiming_uniform_(self.weight_key.data)
        self.weight_query = nn.Parameter(
            torch.zeros(size=(self.unit, self.attention_layer))
        )
        # nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        nn.init.kaiming_uniform_(self.weight_query.data)
        self.weight_attention = nn.Parameter(
            torch.zeros(size=(self.unit * self.attention_layer, self.unit))
        )
        # nn.init.xavier_normal_(self.weight_attention.data, gain=1.414)
        nn.init.kaiming_normal_(self.weight_attention.data)
        self.is_randomwalk_laplacian = is_randomwalk_laplacian
        self.GRU = nn.GRU(self.time_step, self.unit)  # input window, hidden node
        self.cheb_k = 4
        self.multi_layer = multi_layer  # 5
        self.stock_block = nn.ModuleList()
        self.stock_block.extend(
            [
                StockBlockLayer(
                    self.time_step, self.unit, self.multi_layer, dropout_rate, stack_cnt=i
                )
                for i in range(self.stack_cnt)
            ]
        )
        self.fc = nn.Sequential(
            nn.Linear(int(self.time_step), int(self.time_step)),
            nn.LeakyReLU(),
            nn.Linear(int(self.time_step), self.horizon),
        )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.device = device

    def get_laplacian(self, graph, normalize):
        """
        return the laplacian of the graph.
        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(
                torch.mm(D, graph), D
            )
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    def latent_correlation_layer(self, x):
        input, _ = self.GRU(x.permute(2, 0, 1).contiguous())
        input = input.permute(1, 0, 2).contiguous()
        attention = self.self_graph_attention(input)
        attention = torch.mean(attention, dim=0)
        degree = torch.sum(attention, dim=1)
        # laplacian is sym or not
        attention = 0.5 * (attention + attention.T)
        laplacian = calculate_normalized_laplacian(attention)
        # laplacian = calculate_randomwalk_normalized_laplacian(attention)

        mul_L = cheb_polynomial(laplacian, self.cheb_k).to(self.device)

        return mul_L, attention

    def self_graph_attention(self, input):
        input = input.permute(
            0, 2, 1
        ).contiguous()  # batch, node(hidden), node(sequence length, feature)
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)  # batch, node(hidden), 1
        query = torch.matmul(input, self.weight_query)
        data = key.repeat(1, 1, N).view(bat, N * N, -1) + query.repeat(1, N, 1)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = torch.matmul(data, self.weight_attention)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)

        return attention

    def forward(self, x):
        mul_L, attention = self.latent_correlation_layer(x)
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()  # torch.Size([32, 1, 228, 12])
        result = []
        for stack_i in range(self.stack_cnt):
            forecast, X = self.stock_block[stack_i](X, mul_L)
            result.append(forecast)
        forecast = result[0] + result[1]
        forecast = self.fc(forecast)
        if forecast.size()[-1] == 1:
            return forecast.unsqueeze(1).squeeze(-1), attention
        else:
            return forecast.permute(0, 2, 1).contiguous(), attention
