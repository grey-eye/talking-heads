from network.vgg import vgg_face, VGG_Activations
from torchvision.models import vgg19

import torch
from torch import nn
from torch.nn import functional as F

import config


# noinspection PyMethodMayBeStatic
class LossEG(nn.Module):
    def __init__(self, feed_forward=True):
        super(LossEG, self).__init__()

        self.VGG_FACE_AC = VGG_Activations(vgg_face(pretrained=True), [1, 6, 11, 18, 25])
        self.VGG19_AC = VGG_Activations(vgg19(pretrained=True), [1, 6, 11, 20, 29])

        self.IMG_NET_MEAN = torch.Tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
        self.IMG_NET_STD = torch.Tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])

        self.match_loss = not feed_forward

    def loss_cnt(self, x, x_hat):
        IMG_NET_MEAN = torch.Tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
        IMG_NET_STD = torch.Tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])

        x = (x - IMG_NET_MEAN) / IMG_NET_STD
        x_hat = (x_hat - IMG_NET_MEAN) / IMG_NET_STD

        # VGG19 Loss
        vgg19_x_hat = self.VGG19_AC(x_hat)
        vgg19_x = self.VGG19_AC(x)

        vgg19_loss = 0
        for i in range(0, len(vgg19_x)):
            vgg19_loss += F.l1_loss(vgg19_x_hat[i], vgg19_x[i])

        # VGG Face Loss
        vgg_face_x_hat = self.VGG_FACE_AC(x_hat)
        vgg_face_x = self.VGG_FACE_AC(x)

        vgg_face_loss = 0
        for i in range(0, len(vgg_face_x)):
            vgg_face_loss += F.l1_loss(vgg_face_x_hat[i], vgg_face_x[i])

        return vgg19_loss * config.LOSS_VGG19_WEIGHT + vgg_face_loss * config.LOSS_VGG_FACE_WEIGHT

    def loss_adv(self, r_x_hat, d_act, d_act_hat):
        return -r_x_hat.mean() + self.loss_fm(d_act, d_act_hat)

    def loss_fm(self, d_act, d_act_hat):
        loss = 0
        for i in range(0, len(d_act)):
            loss += F.l1_loss(d_act[i], d_act_hat[i])

        return loss

    def loss_mch(self, e_hat, W_i):
        return F.l1_loss(W_i.view(-1), e_hat.view(-1)) * config.LOSS_MCH_WEIGHT

    def forward(self, x, x_hat, r_x_hat, e_hat, W_i, d_act, d_act_hat):
        cnt = self.loss_cnt(x, x_hat)
        adv = self.loss_adv(r_x_hat, d_act, d_act_hat)
        mch = self.loss_mch(e_hat, W_i) if self.match_loss else 0

        return (cnt + adv + mch).view(1)
        # return cnt.view(1), adv.view(1)


class LossD(nn.Module):
    def __init__(self):
        super(LossD, self).__init__()

    def forward(self, r_x, r_x_hat):
        return (F.relu(1 + r_x_hat) + F.relu(1 - r_x)).mean().view(1)
