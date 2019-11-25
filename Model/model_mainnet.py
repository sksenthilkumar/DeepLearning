import torch
import torch.nn as nn
import torchvision.models as models


from Model.model_subnets import *


class MainNet(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.extractor = models.resnet50(pretrained=True)

    def forward(self, input):
        img_features = self.extractor(input)