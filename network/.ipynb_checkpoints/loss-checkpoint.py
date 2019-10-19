from network.vgg import vgg_face, VGG_Activations
from torchvision.models import vgg19 
import torch
from torch import nn
from torch.nn import functional as F

import config


class LossEG(nn.Module):
    def __init__(self, feed_forward=True, gpu=None):
        super(LossEG, self).__init__()

        self.VGG_FACE_AC = VGG_Activations(vgg_face(pretrained=True), [1, 6, 11, 18, 25])
        self.VGG19_AC = VGG_Activations(vgg19(pretrained=True), [1, 6, 11, 20, 29])

        self.match_loss = not feed_forward
        self.gpu = gpu
        if gpu is not None:
            self.cuda(gpu)

    def loss_cnt(self, x, x_hat):
        IMG_NET_MEAN = torch.Tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).to(x.device)
        IMG_NET_STD = torch.Tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).to(x.device)

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

    def loss_adv(self, r_x_hat, d_res, d_res_hat):
        loss_fm = 0
        for res, res_hat in zip(d_res, d_res_hat):
            res = res.cuda(self.gpu)
            res_hat = res_hat.cuda(self.gpu)
            loss_fm += F.l1_loss(res, res_hat)

        return -r_x_hat.mean() + loss_fm * config.LOSS_FM_WEIGHT

    def loss_mch(self, e_hat, W_i):
        return F.l1_loss(W_i.reshape(-1), e_hat.reshape(-1)) * config.LOSS_MCH_WEIGHT

    def forward(self, x, x_hat, r_x_hat, d_res_hat, d_res, e_hat, W_i):
        if self.gpu is not None:
            x = x.cuda(self.gpu)
            x_hat = x_hat.cuda(self.gpu)
            r_x_hat = r_x_hat.cuda(self.gpu)
            e_hat = e_hat.cuda(self.gpu)
            W_i = W_i.cuda(self.gpu)

        cnt = self.loss_cnt(x, x_hat)
        adv = self.loss_adv(r_x_hat, d_res, d_res_hat)
        mch = self.loss_mch(e_hat, W_i) if self.match_loss else 0

        return (cnt + adv + mch).reshape(1)


class LossD(nn.Module):
    def __init__(self, gpu=None):
        super(LossD, self).__init__()
        self.gpu = gpu
        if gpu is not None:
            self.cuda(gpu)

    def forward(self, r_x, r_x_hat):
        if self.gpu is not None:
            r_x = r_x.cuda(self.gpu)
            r_x_hat = r_x_hat.cuda(self.gpu)
        return (F.relu(1 + r_x_hat) + F.relu(1 - r_x)).mean().reshape(1)
