from network.vgg import vgg_face, VGG_Activations
from torchvision.models import vgg19

import torch
from torch import nn
from torch.nn import functional as F

import config


# noinspection PyMethodMayBeStatic
class LossEG(nn.Module):
    def __init__(self, device, feed_forward=True):
        super(LossEG, self).__init__()
        self.dtype = torch.FloatTensor if device == 'cpu' else torch.cuda.FloatTensor

        self.VGG_FACE_AC = VGG_Activations(vgg_face(pretrained=True), [1, 6, 11, 18, 25]).type(self.dtype)
        self.VGG19_AC = VGG_Activations(vgg19(pretrained=True), [1, 6, 11, 20, 29]).type(self.dtype)

        self.match_loss = not feed_forward

    def loss_cnt(self, x, x_hat):
        # VGG19 Loss
        vgg19_x_hat = self.VGG19_AC(x_hat.unsqueeze(0))
        vgg19_x = self.VGG19_AC(x.unsqueeze(0))

        vgg19_loss = 0
        for i in range(0, len(vgg19_x)):
            vgg19_loss += F.l1_loss(vgg19_x_hat[i], vgg19_x[i])

        # VGG Face Loss
        vgg_face_x_hat = self.VGG_FACE_AC(x_hat.unsqueeze(0))
        vgg_face_x = self.VGG_FACE_AC(x.unsqueeze(0))

        vgg_face_loss = 0
        for i in range(0, len(vgg_face_x)):
            vgg_face_loss += F.l1_loss(vgg_face_x_hat[i], vgg_face_x[i])

        return vgg19_loss * config.LOSS_VGG19_WEIGHT + vgg_face_loss * config.LOSS_VGG_FACE_WEIGHT

    def loss_adv(self, r_x_hat, d_act, d_act_hat):
        return -r_x_hat + self.loss_fm(d_act, d_act_hat)

    def loss_fm(self, d_act, d_act_hat):
        loss = 0
        for i in range(0, len(d_act)):
            loss += F.l1_loss(d_act[i], d_act_hat[i])  # / d_act[i].numel()

        return loss

    def loss_mch(self, e_hat, W_i):
        return F.l1_loss(W_i.view(-1), e_hat.view(-1)) * config.LOSS_MCH_WEIGHT

    def forward(self, x, x_hat, r_x_hat, e_hat, W_i, d_act, d_act_hat):
        cnt = self.loss_cnt(x, x_hat)
        adv = self.loss_adv(r_x_hat, d_act, d_act_hat)
        mch = self.loss_mch(e_hat, W_i) if self.match_loss else 0

        return cnt + adv + mch


class LossD(nn.Module):
    def __init__(self, device):
        super(LossD, self).__init__()
        self.dtype = torch.FloatTensor if device == 'cpu' else torch.cuda.FloatTensor
        self.zero = torch.zeros([1]).type(self.dtype)

    def forward(self, r_x, r_x_hat):
        return F.relu(1 + r_x_hat[0]) + F.relu(1 - r_x[0])
