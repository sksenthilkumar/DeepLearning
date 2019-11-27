import torch
import torch.nn as nn
import torchvision.models as models


from Model.parent_classes import *
from Data.data_loader import MrktAttribute


class MainNet(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.extractor = models.resnet50(pretrained=True)
        att_class = MrktAttribute()  # attributes class
        self.atts = dict()
        idxs = 0
        for k, v in att_class.actual_atts.items():
            if k == 'age':
                self.atts[k] = {'op_neur': 4, 'label_idx': (idxs, idxs+v)}
            else:
                self.atts[k] = {'op_neur': v, 'label_idx': (idxs, idxs+v)}
            idxs = idxs + v
            self.atts[k]['model'] = SubNet(op_neurons=self.atts[k]['op_neur'], attr_name=k, **kwargs)
        # self.atts = {'u_body_clothing': {'op_neur': 8, 'label_idx': ()},
        #              'gender': {'op_neur': 1, 'label_idx': ()},
        #              'hair': {'op_neur': 1, 'label_idx': ()},
        #              'age': {'op_neur': 4, 'label_idx': (0, 1)}}

        # for att, infos in self.atts.items():
        #     infos['model'] = SubNet(op_neurons=infos['op_neur'], attr_name=att, **kwargs)

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
        for attr, out_value in op.items():
            model = self.atts[attr]['model']
            strt_idx, end_idx = self.atts[attr]['label_idx']
            label = gt[:, strt_idx:end_idx].squeeze(1)
            loss = loss + model.loss(out_value, label)