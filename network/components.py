""" In this file, PyTorch modules are defined to be used in the Talking Heads model. """

import torch
import torch.nn as nn
from torch.nn import functional as F


def init_conv(conv):
    nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


# region General Blocks

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.rand(1).normal_(0.0, 0.02))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # B: mini batches, C: channels, W: width, H: height
        B, C, H, W = x.shape
        proj_query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(B, -1, W * H)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(B, -1, W * H)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        out = self.gamma * out + x

        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None):
        super(ConvLayer, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride))

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class AdaIn(nn.Module):
    def __init__(self):
        super(AdaIn, self).__init__()
        self.eps = 1e-5

    def forward(self, x, mean_style, std_style):
        B, C, H, W = x.shape

        feature = x.view(B, C, -1)

        std_feat = (torch.std(feature, dim=2) + self.eps).view(B, C, 1)
        mean_feat = torch.mean(feature, dim=2).view(B, C, 1)

        adain = std_style * (feature - mean_feat) / std_feat + mean_style

        adain = adain.view(B, C, H, W)
        return adain


# endregion

# region Non-Adaptive Residual Blocks

class ResidualBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super(ResidualBlockDown, self).__init__()

        # Right Side
        self.conv_r1 = ConvLayer(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_r2 = ConvLayer(out_channels, out_channels, kernel_size, stride, padding)

        # Left Side
        self.conv_l = ConvLayer(in_channels, out_channels, 1, 1)

    def forward(self, x):
        residual = x

        # Right Side
        out = F.relu(x)
        out = self.conv_r1(out)
        out = F.relu(out)
        out = self.conv_r2(out)
        out = F.avg_pool2d(out, 2)

        # Left Side
        residual = self.conv_l(residual)
        residual = F.avg_pool2d(residual, 2)

        # Merge
        out = residual + out
        return out


class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, upsample=2):
        super(ResidualBlockUp, self).__init__()

        # General
        self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')

        # Right Side
        self.norm_r1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.conv_r1 = ConvLayer(in_channels, out_channels, kernel_size, stride)

        self.norm_r2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv_r2 = ConvLayer(out_channels, out_channels, kernel_size, stride)

        # Left Side
        self.conv_l = ConvLayer(in_channels, out_channels, 1, 1)

    def forward(self, x):
        residual = x

        # Right Side
        out = self.norm_r1(x)
        out = F.relu(out)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = self.norm_r2(out)
        out = F.relu(out)
        out = self.conv_r2(out)

        # Left Side
        residual = self.upsample(residual)
        residual = self.conv_l(residual)

        # Merge
        out = residual + out
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.in1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.in2(out)

        out = out + residual
        return out


# endregion

# region Adaptive Residual Blocks


class AdaptiveResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, upsample=2):
        super(AdaptiveResidualBlockUp, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # General
        self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')

        # Right Side
        self.norm_r1 = AdaIn()
        self.conv_r1 = ConvLayer(in_channels, out_channels, kernel_size, stride)

        self.norm_r2 = AdaIn()
        self.conv_r2 = ConvLayer(out_channels, out_channels, kernel_size, stride)

        # Left Side
        self.conv_l = ConvLayer(in_channels, out_channels, 1, 1)

    def forward(self, x, mean1, std1, mean2, std2):
        residual = x

        # Right Side
        out = self.norm_r1(x, mean1, std1)
        out = F.relu(out)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = self.norm_r2(out, mean2, std2)
        out = F.relu(out)
        out = self.conv_r2(out)

        # Left Side
        residual = self.upsample(residual)
        residual = self.conv_l(residual)

        # Merge
        out = residual + out
        return out


class AdaptiveResidualBlock(nn.Module):
    def __init__(self, channels):
        super(AdaptiveResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = AdaIn()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = AdaIn()

    def forward(self, x, mean1, std1, mean2, std2):
        residual = x

        out = self.conv1(x)
        out = self.in1(out, mean1, std1)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.in2(out, mean1, std1)

        out = out + residual
        return out

# endregion
