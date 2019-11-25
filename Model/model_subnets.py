import torch
import torch.nn as nn
import torchvision.models as models

from Model.parent_classes import BaseNet


class ColorNet(BaseNet):
    def __init__(self, op_colors, **kwargs):
        BaseNet.__init__(self)
        self.name = "ColorNet"
        self.no_color = op_colors
        self.in_features = kwargs.get('extractor_op_features')

        # layers
        self.fc1 = nn.Sequential(nn.Linear(self.in_features, 512), nn.Dropout(0.5))
        self.op_layer = nn.Linear(512 ,self.no_color)

    def forward(self, input):
        """
        The function to be executed during training the model
        :param input: batch of image features
        :param debug: return also the intermediate parameters
        :return:
        """
        batch_size = input.size(0)

        out = self.fc1(input)
        out = self.op_layer(out)

        return out
