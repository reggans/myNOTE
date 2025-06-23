import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm.auto import tqdm

from ResNet import ResNet18
from iabn import convert_iabn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NOTE:
    def __init__(self, source_dataloader, target_dataloader, train_config, save_path,
                 iabn=True, alpha=4, bn_momentum=0.1,):
        assert (source_dataloader.dataset.num_classes == target_dataloader.dataset.num_classes)

        self.device = DEVICE
        self.source_dataloader = source_dataloader
        self.target_dataloader = target_dataloader
        self.train_config = train_config
        self.num_classes = source_dataloader.dataset.num_classes

        self.save_path = save_path
        self.iabn = iabn
        self.alpha = alpha
        self.bn_momentum = bn_momentum

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
                module.momentum = self.bn_momentum

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

        # Init train config
        if self.train_config["method"] == "src":
            self.optimizer = torch.optim.SGD(
                self.net.parameters(),
                lr=self.train_config["lr"],
                momentum=self.train_config["momentum"],
                weight_decay=self.train_config["weight_decay"],
                nesterov=True,
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                        T_max=self.train_config["epochs"] * len(self.source_dataloader))
        elif self.train_config["method"] == "NOTE":
            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=self.train_config["lr"],
                weight_decay=self.train_config["weight_decay"],
            )
            self.scheduler = None
        self.class_criterion = torch.nn.CrossEntropyLoss()

        # TODO memory

        self.json = {}
        self.l2_distance = []
        self.occurred_class = [0 for i in range(self.num_classes)]

    def train(self):
        self.net.train()

        class_loss_sum = 0
        total_n = 0
        for data in tqdm(self.source_dataloader, total=len(self.source_dataloader)):
            feats, cls, _ = data
            feats = feats.to(self.device)
            cls = cls.to(self.device)

            preds = self.net(feats)

            class_loss = self.class_criterion(preds, cls)
            class_loss_sum += float(class_loss.item() * feats.shape[0])
            total_n += feats.shape[0]

            self.optimizer.zero_grad()
            class_loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

        avg_loss = class_loss_sum / total_n
        return avg_loss

if __name__ == "__main__":
    pass