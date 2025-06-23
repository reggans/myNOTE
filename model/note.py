from asyncio import current_task

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from ResNet import ResNet18
from iabn import convert_iabn
from utils import memory
from utils.loss_functions import HLoss

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NOTE:
    def __init__(self, source_dataloader, target_dataset, train_config, online_config, save_path,
                 iabn=True, alpha=4, bn_momentum=0.1,
                 memory_type="PBRS", capacity=64,):
        assert (source_dataloader.dataset.num_classes == target_dataset.num_classes)

        # Data-related
        self.device = DEVICE
        self.source_dataloader = source_dataloader
        self.target_dataset = target_dataset
        self.num_classes = source_dataloader.dataset.num_classes

        # Train-related
        self.train_config = train_config
        self.online_config = online_config
        self.save_path = save_path

        # IABN-related
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

        # Manage memory
        self.fifo = memory.FIFO(capacity=capacity)
        if memory_type == "PBRS":
            self.mem = memory.PBRS(capacity=capacity, num_class=self.num_classes)
        elif memory_type == "FIFO":
            self.mem = memory.FIFO(capacity=capacity)
        else:
            raise NotImplementedError

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

    def train_online(self, sample_num, adapt=True):
        TRAINED = 0
        SKIPPED = 1
        FINISHED = 2

        if not hasattr(self, "previous_train_loss"):
            self.previous_train_loss = 0

        N = len(self.target_dataset)
        if sample_num > N:
            return FINISHED

        current_sample =  self.target_dataset[sample_num - 1]
        self.fifo.add_instance(current_sample)

        with torch.no_grad():
            self.net.eval()

            if isinstance(self.mem, memory.FIFO):
                self.mem.add_instance(current_sample)
            else:
                f, c, d = current_sample
                f = f.to(self.device)
                c = c.to(self.device)
                d = d.to(self.device)

                logit = self.net(f.unsqueeze(0))
                pseudo_cls = torch.argmax(logit, keepdim=False)[1][0]
                self.mem.add_instance((f, pseudo_cls, d))

        if (sample_num % self.online_config["update_interval"] != 0
             and not (sample_num == len(self.target_dataset)
                      and self.online_config["update_interval"] >= sample_num)):
                return SKIPPED

        if not self.online_config["use_learned_stats"]:
            self.evaluation_online(sample_num, self.fifo.get_memory())

        if not adapt:
            return TRAINED

        self.net.train()

        if len(self.target_dataset) == 1:
            self.net.eval()

        feats, _, _ = self.mem.get_memory()
        feats = torch.stack(feats)
        dataset = torch.utils.data.TensorDataset(feats)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.train_config["batch_size"],
            shuffle=True,
            drop_last=False,
            pin_memory=False,
        )
        entropy_loss = HLoss(temp_factor=self.online_config["temp_factor"])

        for epoch in range(self.train_config["epochs"]):
            for feats in enumerate(data_loader):
                feats = feats.to(self.device)
                preds = self.net(feats)

                if self.online_config["optimize"]:
                    loss = entropy_loss(preds)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        return TRAINED

    def evaluation_online(self, epoch, current_sample):
        self.net.eval()

        with torch.no_grad():
            feats, cls, do = current_sample

            feats, cls, do = torch.stack(feats), torch.tensor(cls), torch.tensor(do)
            feats, cls, do = feats.to(self.device), cls.to(self.device), do.to(self.device)

            y_pred = self.net(feats).max(1, keepdim=False)[1]

            try:
                true_cls_list = self.json['gt']
                pred_cls_list = self.json['pred']
                accuracy_list = self.json['accuracy']
                f1_macro_list = self.json['f1_macro']
                distance_l2_list = self.json['distance_l2']
            except KeyError:
                true_cls_list = []
                pred_cls_list = []
                accuracy_list = []
                f1_macro_list = []
                distance_l2_list = []

            # append values to lists
            true_cls_list += [int(c) for c in cls]
            pred_cls_list += [int(c) for c in y_pred.tolist()]
            cumul_accuracy = sum(1 for gt, pred in zip(true_cls_list, pred_cls_list) if gt == pred) / float(
                len(true_cls_list)) * 100
            accuracy_list.append(cumul_accuracy)
            f1_macro_list.append(f1_score(true_cls_list, pred_cls_list,
                                          average='macro'))

            self.occurred_class = [0 for i in range(self.num_classes)]

            progress_checkpoint = [int(i * (len(self.target_dataset) / 100.0)) for i in range(1, 101)]
            for i in range(epoch + 1 - len(current_sample[0]), epoch + 1):  # consider a batch input
                if i in progress_checkpoint:
                    print(
                        f'[Online Eval][NumSample:{i}][Epoch:{progress_checkpoint.index(i) + 1}][Accuracy:{cumul_accuracy}]')

            # update self.json file
            self.json = {
                'gt': true_cls_list,
                'pred': pred_cls_list,
                'accuracy': accuracy_list,
                'f1_macro': f1_macro_list,
                'distance_l2': distance_l2_list,
            }

if __name__ == "__main__":
    model = NOTE()