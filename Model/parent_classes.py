import torch
import torch.nn as nn


class SubNet(torch.nn.Module):
    def __init__(self, op_neurons, attr_name, **kwargs):
        """
        :param op_neurons: total number of output neurons
        :param attr_name: subent corresponds to which attribute of market 1501 dataset
        :param kwargs: all additional model parameters
        """
        super().__init__()
        self.in_features = kwargs.get('extractor_op_features')
        self.op_neurons = op_neurons
        self.attr_name = attr_name
        # layers
        self.fc1 = nn.Sequential(nn.Linear(self.in_features, 512), nn.Dropout(0.5))
        self.op_layer = nn.Linear(512, self.op_neurons)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor):
        """
        The function to be executed during training the model
        :param input: mini-batch of image features
        :return:
        """
        out = self.fc1(input)
        out = self.op_layer(out)
        out = self.sigmoid(out)
        return out

    def initialize(self):

        raise NotImplementedError

    def loss(self, op, label):
        """
        :param op: output of the model
        :param label: groud truth
        :return:
        """
        if self.op_neurons == 1:
            return nn.BCELoss(op, label)
        elif self.op_neurons > 1:
            return nn.CrossEntropyLoss(op, label)
        else:
            raise ValueError("Number of outputs cannot be zero")