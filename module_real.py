import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import *

import torch.nn.functional as F

class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1) // 2, bias=False)

    def forward(self, x):
        # x: [B, T, C] (이전에서 transpose를 통해 조정된 크기)
        #print("Input shape to ECA:", x.shape)  # 입력 텐서의 크기 출력
        
        # Apply global average pooling: [B, T, C] -> [B, C, 1]
        y = self.avg_pool(x.transpose(-1, -2))  
        #print("After avg_pool shape:", y.shape)  # Global Average Pooling 후의 크기 출력
        
        # Apply conv1d over the channel dimension: [B, C, 1] -> [B, C, 1]
        y = self.conv(y.squeeze(-1).unsqueeze(1))  # [B, C, 1] -> [B, 1, C] -> [B, 1, C]
        #print("After conv shape:", y.shape)  # Conv1d 후의 크기 출력
        
        # Apply sigmoid activation and multiply
        y = torch.sigmoid(y.squeeze(1))  # [B, 1, C] -> [B, C]
        y = y.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, C] -> [B, 1, C] -> [B, T, C]
        return x * y
    
class AttentiveUConvBlock(nn.Module):
    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4, n_heads=4, att_dims=256, att_dropout=0.1):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList([DilatedConvNorm(in_channels, in_channels, kSize=5, stride=1, groups=in_channels, d=1)])
        
        for i in range(1, upsampling_depth):
            stride = 2 if i > 0 else 1
            self.spp_dw.append(DilatedConvNorm(in_channels, in_channels, kSize=2 * stride + 1, stride=stride, groups=in_channels, d=1))
        
        self.upsampler = torch.nn.Upsample(scale_factor=2) if upsampling_depth > 1 else None
        self.final_norm = NormAct(in_channels)
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
        self.attention = MHANormLayer(in_dim=in_channels, att_dim=att_dims, num_heads=n_heads, dropout=att_dropout)

    def forward(self, x):
        residual = x.clone()
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        
        for k in range(1, self.depth - 1):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        att_in = self.spp_dw[self.depth - 1](output[-1])
        output.append(self.attention(att_in))

        for _ in range(self.depth - 1):
            resampled_out_k = self.upsampler(output.pop(-1))
            output[-1] = output[-1] + resampled_out_k

        expanded = self.final_norm(output[-1])
        return self.res_conv(expanded) + residual

class BSRNN_ECA_UConv(nn.Module):
    def __init__(self, num_channel=128, num_layer=6, num_heads=8, upsampling_depth=4):
        super(BSRNN_ECA_UConv, self).__init__()
        self.num_layer = num_layer
        self.num_channel = num_channel
        self.band_split = BandSplit(channels=num_channel)
        
        self.uconv_blocks = nn.ModuleList([
            AttentiveUConvBlock(
                out_channels=num_channel, 
                in_channels=num_channel, 
                upsampling_depth=upsampling_depth, 
                n_heads=num_heads
            ) for _ in range(self.num_layer)
        ])
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=num_channel, nhead=num_heads, batch_first=True),
            num_layers=num_layer
        )
        
        self.eca_layers = nn.ModuleList([ECA(num_channel) for _ in range(self.num_layer)])
        self.mask_decoder = MaskDecoder(channels=num_channel)

    def forward(self, x):
        x = torch.view_as_real(x)
        z = self.band_split(x).transpose(1, 2)  # [B, N, T, K] 형태

        # 각 대역을 개별적으로 처리하기 위해 대역별로 반복
        outputs = []
        for i in range(z.size(-1)):  # 각 대역별로 반복
            z_band = z[..., i]  # [B, N, T] 형태로 추출
            for j in range(self.num_layer):
                z_band = self.uconv_blocks[j](z_band)
                z_band = self.eca_layers[j](z_band)
            outputs.append(z_band)
        
        # 대역별 출력을 다시 결합
        z = torch.stack(outputs, dim=-1)  # [B, N, T, K] 형태로 복원

        B, N, T, K = z.shape
        z = z.permute(0, 2, 3, 1).reshape(B * T, K * N // self.num_channel, self.num_channel)
        z = self.transformer_encoder(z)
        z = z.reshape(B, T, K, N).permute(0, 3, 1, 2)
        
        m = self.mask_decoder(z)
        m = torch.view_as_complex(m)
        x = torch.view_as_complex(x)
        
        s = m[:, :, 1:-1, 0] * x[:, :, :-2] + m[:, :, 1:-1, 1] * x[:, :, 1:-1] + m[:, :, 1:-1, 2] * x[:, :, 2:]
        s_f = m[:, :, 0, 1] * x[:, :, 0] + m[:, :, 0, 2] * x[:, :, 1]
        s_l = m[:, :, -1, 0] * x[:, :, -2] + m[:, :, -1, 1] * x[:, :, -1]
        s = torch.cat((s_f.unsqueeze(2), s, s_l.unsqueeze(2)), dim=2)
        
        return s

    
class BSRNN(nn.Module):
    def __init__(self, num_channel=128, num_layer=6, num_heads=8):
        super(BSRNN, self).__init__()
        self.num_layer = num_layer
        self.num_channel = num_channel
        self.band_split = BandSplit(channels=num_channel)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=num_channel, nhead=num_heads, batch_first=True),
            num_layers=num_layer
        )

        self.eca_layers_t = nn.ModuleList([ECA(num_channel) for _ in range(self.num_layer)])
        self.eca_layers_k = nn.ModuleList([ECA(num_channel) for _ in range(self.num_layer)])

        for i in range(self.num_layer):
            setattr(self, 'norm_t{}'.format(i + 1), nn.GroupNorm(1, num_channel))
            setattr(self, 'lstm_t{}'.format(i + 1), nn.LSTM(num_channel, 2*num_channel, batch_first=True))
            setattr(self, 'fc_t{}'.format(i + 1), nn.Linear(2*num_channel, num_channel))

        for i in range(self.num_layer):
            setattr(self, 'norm_k{}'.format(i + 1), nn.GroupNorm(1, num_channel))
            setattr(self, 'lstm_k{}'.format(i + 1), nn.LSTM(num_channel, 2*num_channel, batch_first=True, bidirectional=True))
            setattr(self, 'fc_k{}'.format(i + 1), nn.Linear(4*num_channel, num_channel))

        self.mask_decoder = MaskDecoder(channels=num_channel)

        # Perform initialization
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
            if isinstance(m, nn.Linear):
                m.weight.data = torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = torch.view_as_real(x)
        z = self.band_split(x).transpose(1, 2)

        B, N, T, K = z.shape

        skip = z
        for i in range(self.num_layer):
            out = getattr(self, 'norm_t{}'.format(i + 1))(skip)
            out = out.transpose(1, 3).reshape(B*K, T, N)
            out, _ = getattr(self, 'lstm_t{}'.format(i + 1))(out)
            out = getattr(self, 'fc_t{}'.format(i + 1))(out)
            out = self.eca_layers_t[i](out)  # Apply ECA after LSTM
            out = out.reshape(B, K, T, N).transpose(1, 3)
            skip = skip + out

        # Transformer Encoder
        skip = skip.permute(0, 2, 3, 1).reshape(B * T, K * N // self.num_channel, self.num_channel)
        skip = self.transformer_encoder(skip)
        skip = skip.reshape(B, T, K, N).permute(0, 3, 1, 2)

        for i in range(self.num_layer):
            out = getattr(self, 'norm_k{}'.format(i + 1))(skip)
            out = out.permute(0, 2, 3, 1).contiguous().reshape(B*T, K, N)
            out, _ = getattr(self, 'lstm_k{}'.format(i + 1))(out)
            out = getattr(self, 'fc_k{}'.format(i + 1))(out)
            out = self.eca_layers_k[i](out)  # Apply ECA after LSTM
            out = out.reshape(B, T, K, N).permute(0, 3, 1, 2).contiguous()
            skip = skip + out

        m = self.mask_decoder(skip)
        m = torch.view_as_complex(m)
        x = torch.view_as_complex(x)

        s = m[:, :, 1:-1, 0] * x[:, :, :-2] + m[:, :, 1:-1, 1] * x[:, :, 1:-1] + m[:, :, 1:-1, 2] * x[:, :, 2:]
        s_f = m[:, :, 0, 1] * x[:, :, 0] + m[:, :, 0, 2] * x[:, :, 1]
        s_l = m[:, :, -1, 0] * x[:, :, -2] + m[:, :, -1, 1] * x[:, :, -1]
        s = torch.cat((s_f.unsqueeze(2), s, s_l.unsqueeze(2)), dim=2)

        return s

class BandSplit(nn.Module):
    def __init__(self, channels=128):
        super(BandSplit, self).__init__()
        self.band = torch.Tensor([
            2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
            16, 16, 16, 16, 16, 16, 16, 17
        ])
        self.fc_layers = nn.ModuleList([
            nn.Linear(int(band*2), channels) for band in self.band
        ])
        self.norm_layers = nn.ModuleList([
            nn.GroupNorm(1, int(band*2)) for band in self.band
        ])

    def forward(self, x):
        hz_band = 0
        x = x.transpose(1, 2)
        for i, band in enumerate(self.band):
            x_band = x[:, :, hz_band:hz_band+int(band), :]
            x_band = torch.reshape(x_band, [x_band.size(0), x_band.size(1), x_band.size(2)*x_band.size(3)])
            out = self.norm_layers[i](x_band.transpose(1, 2))
            out = self.fc_layers[i](out.transpose(1, 2))

            if i == 0:
                z = out.unsqueeze(3)
            else:
                z = torch.cat((z, out.unsqueeze(3)), dim=3)
            hz_band = hz_band + int(band)
        return z

class MaskDecoder(nn.Module):
    def __init__(self, channels=128):
        super(MaskDecoder, self).__init__()
        self.band = torch.Tensor([
            2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
            16, 16, 16, 16, 16, 16, 16, 17
        ])
        self.fc1_layers = nn.ModuleList([
            nn.Linear(channels, 4*channels) for _ in self.band
        ])
        self.tanh_layers = nn.ModuleList([
            nn.Tanh() for _ in self.band
        ])
        self.fc2_layers = nn.ModuleList([
            nn.Linear(4*channels, int(band*12)) for band in self.band
        ])
        self.glu_layers = nn.ModuleList([
            nn.GLU() for _ in self.band
        ])
        self.norm_layers = nn.ModuleList([
            nn.GroupNorm(1, channels) for _ in self.band
        ])

    def forward(self, x):
        for i, band in enumerate(self.band):
            x_band = x[:, :, :, i]
            out = self.norm_layers[i](x_band)
            out = self.fc1_layers[i](out.transpose(1, 2))
            out = self.tanh_layers[i](out)
            out = self.fc2_layers[i](out)
            out = self.glu_layers[i](out)
            out = torch.reshape(out, [out.size(0), out.size(1), int(out.size(2)/6), 3, 2])
            if i == 0:
                m = out
            else:
                m = torch.cat((m, out), dim=2)
        return m.transpose(1, 2)

class Discriminator(nn.Module):
    def __init__(self, ndf, in_channel=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, ndf, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf*2, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf*2, affine=True),
            nn.PReLU(2*ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf*2, ndf*4, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf*4, affine=True),
            nn.PReLU(4*ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf*4, ndf*8, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf*8, affine=True),
            nn.PReLU(8*ndf),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf*8, ndf*4)),
            nn.Dropout(0.3),
            nn.PReLU(4*ndf),
            nn.utils.spectral_norm(nn.Linear(ndf*4, 1)),
            LearnableSigmoid(1)
        )

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.layers(xy)
    
"""!
@brief Attentive SuDO-RM-RF model

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import torch
import torch.nn as nn
import math


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size),
                                  requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size),
                                 requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, channels, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) +
                self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x):
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + 1e-8).sqrt())


class ConvNormAct(nn.Module):
    '''
    This class defines the convolution layer with normalization and a PReLU
    activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=True, groups=groups)
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    '''
    This class defines the convolution layer with normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=True, groups=groups)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class NormAct(nn.Module):
    '''
    This class defines a normalization and PReLU activation
    '''
    def __init__(self, nOut):
        '''
        :param nOut: number of output channels
        '''
        super().__init__()
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.norm(input)
        return self.act(output)


class DilatedConv(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d,
                              padding=((kSize - 1) // 2) * d, groups=groups)

    def forward(self, input):
        return self.conv(input)


class DilatedConvNorm(nn.Module):
    '''
    This class defines the dilated convolution with normalized output.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d,
                              padding=((kSize - 1) // 2) * d, groups=groups)
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class MHANormLayer(nn.Module):
    """Multi-head attention with residual addition and normalization."""
    def __init__(self, in_dim, att_dim=128, num_heads=4, dropout=0.1, max_len=5000):
        super(MHANormLayer, self).__init__()
        self.mha = nn.MultiheadAttention(
            att_dim, num_heads=num_heads, dropout=dropout,
            bias=True, add_bias_kv=False,
            add_zero_attn=False, kdim=None, vdim=None, batch_first=True
        )
        # 입력을 어텐션 차원(att_dim)으로 변환하는 선형 레이어
        self.in_linear = nn.Linear(in_dim, att_dim)
        self.in_norm = GlobLN(att_dim)
        self.out_norm1 = GlobLN(att_dim)
        self.out_norm2 = GlobLN(in_dim)
        # 어텐션 결과를 다시 원래 차원으로 변환
        self.out_linear = nn.Linear(att_dim, in_dim)
        self.pos_enc = PositionalEncoding(d_model=att_dim, dropout=dropout, max_len=max_len)
        self.act = nn.PReLU()

    def forward(self, x):
        # 입력을 어텐션 차원으로 변환하고, 포지셔널 인코딩을 적용
        x = self.in_linear(x.transpose(1, 2))
        x = self.pos_enc(x)
        x = self.in_norm(x.transpose(1, 2)).transpose(1, 2)
        
        # 어텐션 적용 후 첫 번째 정규화와 잔차 연결
        x = x + self.out_norm1(
            self.mha(query=x, key=x, value=x)[0].transpose(1, 2)
        ).transpose(1, 2)
        
        # 원래 차원으로 변환한 후 두 번째 정규화와 활성화 함수 적용
        return self.act(self.out_norm2(self.out_linear(x).transpose(1, 2)))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class AttentiveUConvBlock(nn.Module):
    '''
    This class defines the block which performs successive downsampling and
    upsampling in order to be able to analyze the input features in multiple
    resolutions. Moreover, it defines an attention layer which is used for
    better sequence modeling at the most downsampled level.
    '''

    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 upsampling_depth=4,
                 n_heads=4,
                 att_dims=256,
                 att_dropout=0.1):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1,
                                    stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(DilatedConvNorm(in_channels, in_channels, kSize=5,
                                           stride=1, groups=in_channels, d=1))

        for i in range(1, upsampling_depth):
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.spp_dw.append(DilatedConvNorm(in_channels, in_channels,
                                               kSize=2*stride + 1,
                                               stride=stride,
                                               groups=in_channels, d=1))
        if upsampling_depth > 1:
            self.upsampler = torch.nn.Upsample(scale_factor=2,
                                               # align_corners=True,
                                               # mode='bicubic'
                                               )
        self.final_norm = NormAct(in_channels)
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)

        # Attention layer
        self.attention = MHANormLayer(in_dim=in_channels,
                                      att_dim=att_dims,
                                      num_heads=n_heads,
                                      dropout=att_dropout)

    def forward(self, x):
        residual = x.clone()
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        
        for k in range(1, self.depth - 1):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        att_in = self.spp_dw[self.depth - 1](output[-1])
        output.append(self.attention(att_in))

        # Gather in reverse order and match dimensions with the original tensor
        for _ in range(self.depth - 1):
            resampled_out_k = self.upsampler(output.pop(-1))
            # 원래 텐서와 업샘플링된 텐서의 크기가 일치하지 않는 경우 크기 조정
            if resampled_out_k.size(2) != output[-1].size(2):
                resampled_out_k = resampled_out_k[:, :, :output[-1].size(2)]
            output[-1] = output[-1] + resampled_out_k

        expanded = self.final_norm(output[-1])
        return self.res_conv(expanded) + residual


class SuDORMRF(nn.Module):
    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 num_blocks=16,
                 upsampling_depth=4,
                 enc_kernel_size=21,
                 enc_num_basis=512,
                 n_heads=4,
                 att_dims=256,
                 att_dropout=0.1,
                 num_sources=2):
        super(SuDORMRF, self).__init__()

        # Number of sources to produce
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size
        self.enc_num_basis = enc_num_basis
        self.num_sources = num_sources

        # Appropriate padding is needed for arbitrary lengths
        self.lcm = abs(self.enc_kernel_size // 2 * 2 **
                       self.upsampling_depth) // math.gcd(
                       self.enc_kernel_size // 2,
                       2 ** self.upsampling_depth)

        # Front end
        self.encoder = nn.Conv1d(in_channels=1, out_channels=enc_num_basis,
                                 kernel_size=enc_kernel_size,
                                 stride=enc_kernel_size // 2,
                                 padding=enc_kernel_size // 2,
                                 bias=False)
        torch.nn.init.xavier_uniform(self.encoder.weight)

        # Norm before the rest, and apply one more dense layer
        self.ln = GlobLN(enc_num_basis)
        self.bottleneck = nn.Conv1d(
            in_channels=enc_num_basis,
            out_channels=out_channels,
            kernel_size=1)

        # Separation module
        self.sm = nn.Sequential(*[
            AttentiveUConvBlock(out_channels=out_channels,
                                in_channels=in_channels,
                                upsampling_depth=upsampling_depth,
                                n_heads=n_heads,
                                att_dims=att_dims,
                                att_dropout=att_dropout)
            for _ in range(num_blocks)])

        mask_conv = nn.Conv1d(out_channels, num_sources * enc_num_basis, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        # Back end
        self.decoder = nn.ConvTranspose1d(
            in_channels=enc_num_basis * num_sources,
            out_channels=num_sources,
            output_padding=(enc_kernel_size // 2) - 1,
            kernel_size=enc_kernel_size,
            stride=enc_kernel_size // 2,
            padding=enc_kernel_size // 2,
            groups=1, bias=False)
        torch.nn.init.xavier_uniform(self.decoder.weight)
        self.mask_nl_class = nn.ReLU()
    # Forward pass
    def forward(self, input_wav):
        # Front end
        x = self.pad_to_appropriate_length(input_wav)
        x = self.encoder(x)

        # Split paths
        s = x.clone()
        # Separation module
        x = self.ln(x)
        x = self.bottleneck(x)
        x = self.sm(x)

        x = self.mask_net(x)
        x = x.view(x.shape[0], self.num_sources, self.enc_num_basis, -1)
        x = self.mask_nl_class(x)
        x = x * s.unsqueeze(1)
        # Back end
        estimated_waveforms = self.decoder(x.view(x.shape[0], -1, x.shape[-1]))
        return self.remove_trailing_zeros(estimated_waveforms, input_wav)

    def pad_to_appropriate_length(self, x):
        values_to_pad = int(x.shape[-1]) % self.lcm
        if values_to_pad:
            appropriate_shape = x.shape
            padded_x = torch.zeros(
                list(appropriate_shape[:-1]) +
                [appropriate_shape[-1] + self.lcm - values_to_pad],
                dtype=torch.float32)
            padded_x[..., :x.shape[-1]] = x
            return padded_x
        return x

    @staticmethod
    def remove_trailing_zeros(padded_x, initial_x):
        return padded_x[..., :initial_x.shape[-1]]


if __name__ == "__main__":

    from time import time
    model = SuDORMRF(out_channels=256,
                     in_channels=512,
                     num_blocks=8,
                     upsampling_depth=5,
                     enc_kernel_size=21,
                     enc_num_basis=512,
                     n_heads=3,
                     att_dims=256,
                     att_dropout=0.1,
                     num_sources=4)

    dummy_input = torch.rand(3, 1, 32079)
    now = time()
    estimated_sources = model(dummy_input)
    print(estimated_sources.shape, time() - now)




