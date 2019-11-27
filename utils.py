import yaml
import torch
import pathlib
from Model.model_factory import ModelFactory


class RunningAverage:
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def top1_acc(op, target):
    topk = 1
    batch_size = target.size(0)
    _, pred = op.topk(topk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1))
    correct_k = correct[:topk].view(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(1.0/batch_size)


def load_model_frm_exp(exp_path):
    """
    Loading a model from the given experiment folder path
    """
    exp_path = pathlib.Path(exp_path)
    model_path = exp_path.joinpath('best.pth.tar')
    tc_path = exp_path.joinpath('training_config.yaml')
    tc = yaml.full_load(open(str(tc_path)))
    return load_model(tc, model_path)


def load_model(model_path, tc):
    """
    Loading a model from the given saved path and training config
    """
    saved_model = torch.load(model_path)
    model = ModelFactory(**tc)
    model.load_state_dict(saved_model['model_state_dict'])
    return model
