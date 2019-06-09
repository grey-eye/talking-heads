""" Implementation of the three networks that make up the Talking Heads generative model. """
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

from .components import ResidualBlock, ResidualBlockDown, ResidualBlockUp, SelfAttention
from .adain import adain_direct as adain


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Embedder(nn.Module):
    """
    The Embedder network attempts to generate a vector that encodes the personal characteristics of an individual given
    a head-shot and the matching landmarks.
    """
    def __init__(self):
        super(Embedder, self).__init__()

        # TODO: We need 6 downsize layers, but channels are between 64 and 512, do we stop adding channels after conv4?

        self.conv1 = ResidualBlockDown(6, 64)
        self.conv2 = ResidualBlockDown(64, 128)
        self.conv3 = ResidualBlockDown(128, 256)
        self.att = SelfAttention(256)
        self.conv4 = ResidualBlockDown(256, 512)
        self.conv5 = ResidualBlockDown(512, 512)
        self.conv6 = ResidualBlockDown(512, 512)

        self.pooling = nn.AdaptiveMaxPool2d((1, 1))

        self.apply(weights_init)

    def forward(self, x, y):
        assert list(x.shape) == [3, 224, 224], "Both x and y must be tensors with shape HWC [3, 224, 224]."
        assert x.shape == y.shape, "Both x and y must be tensors with shape HWC [3, 224, 224]."

        # Concatenate x & y and shape them into a [1, 6, 224, 224] array
        out = torch.cat((x, y), dim=0)
        out = out.unsqueeze(0)

        # Encode
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.att(out)
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))

        # Vectorize
        out = F.relu(self.pooling(out).view(512, 1))

        return out


class Generator(nn.Module):
    PSI_WIDTH = 32
    ADAIN_LAYERS = OrderedDict([
        ('deconv4', 256),
        ('deconv3', 128),
        ('deconv2', 64),
        ('deconv1', 3)
    ])

    def __init__(self):
        super(Generator, self).__init__()

        # projection layer
        self.PSI_PORTIONS, psi_length = self.define_psi_slices()
        self.projection = nn.Parameter(torch.rand(psi_length, 512).normal_(0.0, 0.02))

        # encoding layers
        self.conv1 = ResidualBlockDown(3, 64)
        self.in1_e = nn.InstanceNorm2d(64, affine=True)

        self.conv2 = ResidualBlockDown(64, 128)
        self.in2_e = nn.InstanceNorm2d(128, affine=True)

        self.conv3 = ResidualBlockDown(128, 256)
        self.in3_e = nn.InstanceNorm2d(256, affine=True)

        self.att1 = SelfAttention(256)

        self.conv4 = ResidualBlockDown(256, 512)
        self.in4_e = nn.InstanceNorm2d(512, affine=True)

        # residual layers
        self.res1 = ResidualBlock(512)
        self.res2 = ResidualBlock(512)
        self.res3 = ResidualBlock(512)
        self.res4 = ResidualBlock(512)
        self.res5 = ResidualBlock(512)

        # decoding layers
        self.deconv4 = ResidualBlockUp(512, 256, upsample=2)
        self.deconv3 = ResidualBlockUp(256, 128, upsample=2)
        self.att2 = SelfAttention(128)
        self.deconv2 = ResidualBlockUp(128, 64, upsample=2)
        self.deconv1 = ResidualBlockUp(64, 3, upsample=2)

        self.apply(weights_init)

    def forward(self, y, e):
        assert list(y.shape) == [3, 224, 224], "Vector y must have shape HWC [3, 224, 224]"

        # Shape y into a [1, 3, 224, 224] array
        out = y.unsqueeze(0)

        # Calculate psi_hat parameters
        psi_hat = torch.mm(self.projection, e).view(-1)

        # Encode
        out = F.relu(self.in1_e(self.conv1(out)))
        out = F.relu(self.in2_e(self.conv2(out)))
        out = F.relu(self.in3_e(self.conv3(out)))
        out = self.att1(out)
        out = F.relu(self.in4_e(self.conv4(out)))

        # Residual layers
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)

        # Decode
        out = F.relu(adain(self.deconv4(out), *self.slice_psi(psi_hat, 'deconv4')))
        out = F.relu(adain(self.deconv3(out), *self.slice_psi(psi_hat, 'deconv3')))
        out = self.att2(out)
        out = F.relu(adain(self.deconv2(out), *self.slice_psi(psi_hat, 'deconv2')))
        out = torch.tanh(adain(self.deconv1(out), *self.slice_psi(psi_hat, 'deconv1')))

        return out[0]

    # def slice_psi(self, psi, portion):
    #     idx0, idx1 = self.PSI_PORTIONS[portion]
    #     return psi[idx0:idx1].view(1, -1, self.PSI_WIDTH)
    #
    # def define_psi_slices(self):
    #     out = {}
    #     d = self.ADAIN_LAYERS
    #     start_idx, end_idx = 0, 0
    #     for layer in d:
    #         end_idx = start_idx + d[layer] * self.PSI_WIDTH
    #         out[layer] = (start_idx, end_idx)
    #         start_idx = end_idx
    #
    #     return out, end_idx

    def slice_psi(self, psi, portion):
        idx0, idx1 = self.PSI_PORTIONS[portion]
        aux = psi[idx0:idx1].view(2, -1)
        return aux[0].view(1, -1, 1, 1), aux[1].view(1, -1, 1, 1)

    def define_psi_slices(self):
        out = {}
        d = self.ADAIN_LAYERS
        start_idx, end_idx = 0, 0
        for layer in d:
            end_idx = start_idx + d[layer] * 2
            out[layer] = (start_idx, end_idx)
            start_idx = end_idx

        return out, end_idx


class Discriminator(nn.Module):
    def __init__(self, training_videos):
        super(Discriminator, self).__init__()

        self.conv1 = ResidualBlockDown(6, 64)
        self.conv2 = ResidualBlockDown(64, 128)
        self.conv3 = ResidualBlockDown(128, 256)
        self.att = SelfAttention(256)
        self.conv4 = ResidualBlockDown(256, 512)
        self.conv5 = ResidualBlockDown(512, 512)
        self.conv6 = ResidualBlockDown(512, 512)
        self.res_block = ResidualBlock(512)

        self.pooling = nn.AdaptiveMaxPool2d((1, 1))

        self.W = nn.Parameter(torch.rand(512, training_videos).normal_(0.0, 0.02))
        self.w_0 = nn.Parameter(torch.rand(512, 1).normal_(0.0, 0.02))
        self.b = nn.Parameter(torch.rand(1).normal_(0.0, 0.02))

        self.apply(weights_init)

    def forward(self, x, y, i):
        assert list(x.shape) == [3, 224, 224], "Both x and y must be tensors with shape HWC [3, 224, 224]."
        assert x.shape == y.shape, "Both x and y must be tensors with shape HWC [3, 224, 224]."

        # Concatenate x & y and shape them into a [1, 6, 224, 224] array
        out = torch.cat((x, y), dim=0)
        out = out.unsqueeze(0)

        # Encode
        out_0 = F.relu(self.conv1(out))
        out_1 = F.relu(self.conv2(out_0))
        out_2 = F.relu(self.conv3(out_1))
        out_3 = self.att(out_2)
        out_4 = F.relu(self.conv4(out_3))
        out_5 = F.relu(self.conv5(out_4))
        out_6 = F.relu(self.conv6(out_5))
        out_7 = self.res_block(out_6)

        # Vectorize
        out = F.relu(self.pooling(out_7).view(512, 1))

        # Calculate Realism Score
        out = torch.mm(out.t(), self.W[:, i].view(-1, 1) + self.w_0) + self.b
        # out = torch.tanh(out)

        return out, [out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7]
