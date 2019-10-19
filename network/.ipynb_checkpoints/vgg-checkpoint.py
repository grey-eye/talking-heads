import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import vgg

import config


class VGG_Activations(nn.Module):
    """
    This class allows us to execute only a part of a given VGG network and obtain the activations for the specified
    feature blocks. Note that we are taking the result of the activation function after each one of the given indeces,
    and that we consider 1 to be the first index.
    """
    def __init__(self, vgg_network, feature_idx):
        super(VGG_Activations, self).__init__()
        features = list(vgg_network.features)
        self.features = nn.ModuleList(features).eval()
        self.idx_list = feature_idx

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in self.idx_list:
                results.append(x)

        return results


def vgg_face(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = vgg.VGG(vgg.make_layers(vgg.cfgs['D'], batch_norm=False), num_classes=2622, **kwargs)
    if pretrained:
        print("111111111111")
        model.load_state_dict(vgg_face_state_dict())
    return model


def vgg_face_state_dict():
    default = torch.load(config.VGG_FACE)
    state_dict = OrderedDict({
        'features.0.weight': default['conv1_1.weight'],
        'features.0.bias': default['conv1_1.bias'],
        'features.2.weight': default['conv1_2.weight'],
        'features.2.bias': default['conv1_2.bias'],
        'features.5.weight': default['conv2_1.weight'],
        'features.5.bias': default['conv2_1.bias'],
        'features.7.weight': default['conv2_2.weight'],
        'features.7.bias': default['conv2_2.bias'],
        'features.10.weight': default['conv3_1.weight'],
        'features.10.bias': default['conv3_1.bias'],
        'features.12.weight': default['conv3_2.weight'],
        'features.12.bias': default['conv3_2.bias'],
        'features.14.weight': default['conv3_3.weight'],
        'features.14.bias': default['conv3_3.bias'],
        'features.17.weight': default['conv4_1.weight'],
        'features.17.bias': default['conv4_1.bias'],
        'features.19.weight': default['conv4_2.weight'],
        'features.19.bias': default['conv4_2.bias'],
        'features.21.weight': default['conv4_3.weight'],
        'features.21.bias': default['conv4_3.bias'],
        'features.24.weight': default['conv5_1.weight'],
        'features.24.bias': default['conv5_1.bias'],
        'features.26.weight': default['conv5_2.weight'],
        'features.26.bias': default['conv5_2.bias'],
        'features.28.weight': default['conv5_3.weight'],
        'features.28.bias': default['conv5_3.bias'],
        'classifier.0.weight': default['fc6.weight'],
        'classifier.0.bias': default['fc6.bias'],
        'classifier.3.weight': default['fc7.weight'],
        'classifier.3.bias': default['fc7.bias'],
        'classifier.6.weight': default['fc8.weight'],
        'classifier.6.bias': default['fc8.bias']
    })
    return state_dict
