import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ResNet import ResNet18
from iabn import convert_iabn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NOTE:
    def __init__(self, source_dataloader, target_dataloader, save_path,
                 iabn=True, alpha=4, momentum=0.1,):
        assert (source_dataloader.dataset.num_classes == target_dataloader.dataset.num_classes)

        self.device = DEVICE
        self.source_dataloader = source_dataloader
        self.target_dataloader = target_dataloader
        self.num_classes = source_dataloader.dataset.num_classes

        self.save_path = save_path
        self.iabn = iabn
        self.alpha = alpha
        self.momentum = momentum

        # Init net
        self.net = torchvision.models.resnet18(pretrained=True).to(self.device)
        if self.iabn:
            convert_iabn(self.net, self.alpha)
        num_feats = self.net.fc.in_features
        self.net.fc = nn.Linear(num_feats, self.num_classes)

        # Manage net params
        for param in self.net.parameters():
            param.requires_grad = False
        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                module.track_running_stats = True
                module.momentum = self.momentum

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

    def train(self):
        self.net.train()


if __name__ == "__main__":
    pass