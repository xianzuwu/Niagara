# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models


RESNETS = {18: (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
           50: (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
           152: (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V2)}


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50,152], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3],152: [3, 8, 36, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck, 152: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)
    model, weigths = RESNETS[num_layers]

    if pretrained:
        loaded = torch.hub.load_state_dict_from_url(weigths.url)
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, bn_order, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.bn_order = bn_order

        if num_layers not in RESNETS:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            model, weights = RESNETS[num_layers]
            self.encoder = model(weights=weights)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

# niagara encoder
class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder"""
    def __init__(self, num_layers, pretrained, bn_order, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.bn_order = bn_order

        if num_layers not in RESNETS:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            model, weights = RESNETS[num_layers]
            self.encoder = model(weights=weights)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        encoder = self.encoder
        features = []

        # Assume the shape of input_image is [B, C, H, W].
        # where C = 3 (RGB) + 1 (Depth) + 3 * Triplane Channels

        rgb = input_image[:, :3, :, :]        # [B, 3, H, W]
        depth = input_image[:, 3:4, :, :]     # [B, 1, H, W]
        normal = input_image[:, 4:7, :, :]
        triplane = input_image[:, 7:, :, :]   # [B, 3 * num_channels, H, W]

        norm_mean = torch.tensor([0.45] * 3 + [0.5] * 4, device=input_image.device).view(1, 7, 1, 1)
        norm_std = torch.tensor([0.225] * 3 + [0.2] * 4, device=input_image.device).view(1, 7, 1, 1)

        x_rgb_depth_normal = (torch.cat([rgb, depth, normal], dim=1) - norm_mean) / norm_std
        x = torch.cat([x_rgb_depth_normal, triplane], dim=1)

        x = encoder.conv1(x)
        if self.bn_order == "pre_bn":
            features.append(x)
            x = encoder.bn1(x)
            x = encoder.relu(x)
        elif self.bn_order == "monodepth":
            x = encoder.bn1(x)
            x = encoder.relu(x)
            features.append(x)
        else:
            raise ValueError(f"Unsupported bn_order: {self.bn_order}")

        features.append(encoder.layer1(encoder.maxpool(x)))
        features.append(encoder.layer2(features[-1]))
        features.append(encoder.layer3(features[-1]))
        features.append(encoder.layer4(features[-1]))

        return features