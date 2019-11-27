import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models

from Model.model_mainnet import MainNet


class AugNet(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layers = None # to be implemented

    def forward(self, input):
        # passes through layers and selects the best augmentation policy
        return None


class AutoAugNet(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.augnet = AugNet()
        self.mainnet = MainNet()

    def forward(self, input):
        aug_policy = self.augnet(input)
        aug_ip = self.transform(input, aug_policy)
        out = self.mainnet(aug_ip)
        return out

    def transform(self, input, *params):
        # augments the input based on the given augmentation policy
        aug_ip = None
        return aug_ip