import torch
import torch.nn as nn
import torchvision.models as models

from Model.parent_classes import SubNet


class ColorNet(SubNet):
    def __init__(self, op_colors, **kwargs):
        SubNet.__init__(self, **kwargs)
        self.name = "ColorNet"
        self.no_color = op_colors
        # layers
        self.fc1 = nn.Sequential(nn.Linear(self.in_features, 512), nn.Dropout(0.5))
        self.op_layer = nn.Linear(512, self.no_color)

    def forward(self, input: torch.Tensor):
        out = self.fc1(input)
        out = self.op_layer(out)

        return out
