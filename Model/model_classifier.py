import torch
import torchvision.models as models


class DoubleHeadedClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "DoubleHeadedClassifier"
        self.feature_extractor = models.resnet50(pretrained=True)

    def forward(self, input:torch.Tensor, inter_params=False):
        """
        The function to be executed during training the model
        :param input: input tensor
        :param inter_params: return also the intermediate parameters
        :return:
        """
        batch_size = input.size(0)

        if inter_params:
            raise NotImplementedError
        return None
