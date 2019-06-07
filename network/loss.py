from network.vgg import vgg_face, VGG_Activations
from torchvision.models import vgg19

import torch
from torch.nn import functional as F

import config


# noinspection PyMethodMayBeStatic
class TalkingHeadsLoss(object):
    def __init__(self, device):
        self.dtype = torch.FloatTensor if device == 'cpu' else torch.cuda.FloatTensor

        VGG_FACE = vgg_face(pretrained=True)
        self.VGG_FACE_AC = VGG_Activations(VGG_FACE, [1, 6, 11, 18, 25]).type(self.dtype)
        VGG19 = vgg19(pretrained=True)
        self.VGG19_AC = VGG_Activations(VGG19, [1, 6, 11, 20, 29]).type(self.dtype)

    def loss_E_G(self, x, x_hat, r_x_hat, e_hat, W_i, residual_act):
        return \
            self._loss_cnt(x, x_hat) + \
            self._loss_adv(r_x_hat, residual_act) + \
            self._loss_mch(e_hat, W_i)

    def loss_D(self, r_x, r_x_hat):
        return max(torch.zeros([1]), 1 + r_x_hat) + max(torch.zeros([1]), 1 - r_x)

    # region Internal Functions
    def _loss_cnt(self, x, x_hat):
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

    def _loss_adv(self, r_x_hat, residual_act):
        return -r_x_hat + self._loss_fm(residual_act)

    def _loss_fm(self, residual_act):
        # TODO: Figure out how to implement this
        return 0

    def _loss_mch(self, e_hat, W_i):
        # return F.l1_loss(W_i.view(-1), e_hat.view(-1)) * config.LOSS_MCH_WEIGHT
        return 0

    # endregion

