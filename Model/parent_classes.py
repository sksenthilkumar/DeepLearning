import torch


class BaseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        """
        :param input: the output feature of the extractor
        :return:
        """
        raise NotImplementedError

    def initialize(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError