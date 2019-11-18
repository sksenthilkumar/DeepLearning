import torch


class ModelTemplate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = None

    def forward(self, inter_params=False):
        """
        The function to be executed during training the model
        :param inter_params: return also the intermediate parameters
        :return:
        """
        if inter_params:
            raise NotImplementedError
        return None

    def loss(self, ip, target):
        loss = torch.nn.functional.cross_entropy(ip, target)
        return loss


