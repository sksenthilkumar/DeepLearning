import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models

from Data.data_loader import MrktAttribute
import utils


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

        self.metric = utils.top1_acc

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
            op = op.squeeze(1)
            if type(label) is not torch.FloatTensor:
                label = label.type(torch.FloatTensor)
            loss_func = nn.BCELoss()

        elif self.op_neurons > 1:
            loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Number of outputs cannot be zero")

        return loss_func(op, label)


class MainNet(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.extractor = models.resnet50(pretrained=True)
        att_class = MrktAttribute()  # attributes class
        total_op_neurons = att_class.total_op_neurons
        self.atts = dict()
        idxs = 0
        for k, v in att_class.actual_atts.items():
            op_neuron = att_class.atts_op_neurons[k]
            self.atts[k] = {'op_neur': op_neuron, 'label_idx': (idxs, idxs+v),
                            'label_typ': att_class.atts_label_type[k], 'loss_weight': op_neuron/total_op_neurons,
                            'each_class_weight': att_class.atts_weights[k]}
            idxs = idxs + v
            self.atts[k]['model'] = SubNet(op_neurons=self.atts[k]['op_neur'], attr_name=k, **kwargs)

        # fixing the weights of the first 7 layers of resnet-50
        ct = 0
        for child in self.extractor.children():
            ct += 1
        if ct < 7:
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, input):
        img_features = self.extractor(input)
        output = {}
        for att, infos in self.atts.items():
            model = infos['model']
            out = model(img_features)
            output[att] = out

        return output

    def loss(self, op, gt):
        """

        :param op: dict: model output
        :param gt: ground truth
        :return: total loss value
        """
        loss = 0
        loss_dict = {}
        for attr, out_value in op.items():
            model = self.atts[attr]['model']
            loss_weight = self.atts[attr]['loss_weight']
            strt_idx, end_idx = self.atts[attr]['label_idx']
            each_class_weight = self.atts[attr]['each_class_weight']
            label = gt[:, strt_idx:end_idx].squeeze(1)
            if self.atts[attr]['label_typ'] == 'o':
                # it is one hot convert it to integer
                _, label = label.max(dim=1)
            subnet_loss = model.loss(out_value, label, weights=each_class_weight)
            loss_dict[attr] = (subnet_loss, loss_weight)
            loss = loss + (loss_weight * loss_weight)
        loss = torch.autograd.Variable(torch.from_numpy(np.array([loss])), requires_grad=True)
        return loss, loss_dict

    def metric(self, op, gt):
        """

        :param op: dict: model output
        :param gt: ground truth
        :return: average accuracy value
        """
        accs = {}
        for attr, out_value in op.items():
            model = self.atts[attr]['model']
            strt_idx, end_idx = self.atts[attr]['label_idx']
            label = gt[:, strt_idx:end_idx].squeeze(1)
            if self.atts[attr]['label_typ'] == 'o':
                # it is one hot convert it to integer
                _, label = label.max(dim=1)

            accs[attr] = model.metric(out_value, label)
        list_accs = list(accs.values())
        accs['average'] = sum(list_accs)/len(list_accs)
        return accs
