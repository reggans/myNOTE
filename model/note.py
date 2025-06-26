import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from .iabn import convert_iabn, IABN1d, IABN2d
from utils import memory
from utils.loss_functions import HLoss
from utils.logging import to_json
from .ResNet import ResNet18

DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class NOTE:
    def __init__(self, source_dataset, val_dataset, target_dataset,
                 train_config, online_config,
                 save_path, checkpoint_path,
                 iabn=True, alpha=4,
                 memory_type="PBRS", capacity=64,):
        assert (source_dataset.num_classes == target_dataset.num_classes)

        # Data-related
        self.device = DEVICE
        self.source_dataloader = torch.utils.data.DataLoader(source_dataset, batch_size=train_config["batch_size"],)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=train_config["batch_size"],)
        self.target_dataset = target_dataset        # Kept as dataset for ease later
        self.num_classes = source_dataset.num_classes

        # Train-related
        self.train_config = train_config
        self.online_config = online_config
        self.save_path = save_path
        self.checkpoint_path = checkpoint_path

        # IABN-related
        self.iabn = iabn
        self.alpha = alpha
        self.bn_momentum = train_config["bn_momentum"]

        # Init net
        self.net = ResNet18().to(self.device)
        if self.iabn:
            convert_iabn(self.net, self.alpha)
        num_feats = self.net.fc.in_features
        self.net.fc = nn.Linear(num_feats, self.num_classes).to(self.device)

        # Manage net params
        for param in self.net.parameters():
            param.requires_grad = False
        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                module.track_running_stats = True
                module.momentum = self.bn_momentum

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            if isinstance(module, IABN1d) or isinstance(module, IABN2d):
                for param in module.parameters():
                    param.requires_grad = True

        # Init train config
        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.train_config["lr"],
            momentum=self.train_config["momentum"],
            weight_decay=self.train_config["weight_decay"],
            nesterov=True,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    T_max=self.train_config["epochs"] * len(self.source_dataloader))

        # Init online train config
        self.online_optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.online_config["lr"],
            weight_decay=self.online_config["weight_decay"],
        )

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

    def evaluation(self):
        self.net.eval()

        class_loss_sum = 0
        class_acc_sum = 0
        total_n = 0

        for data in self.val_dataloader:
            feats, cls, _ = data
            feats = feats.to(self.device)
            cls = cls.to(self.device)

            with torch.no_grad():
                preds = self.net(feats)
                class_loss = self.class_criterion(preds, cls)

            class_loss_sum += float(class_loss.item() * feats.shape[0])
            class_acc_sum += (preds.max(1, keepdim=False)[1] == cls).sum()
            total_n += feats.shape[0]

        return class_loss_sum / total_n, class_acc_sum / total_n

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
                pseudo_cls = logit.max(1, keepdim=False)[1][0]
                self.mem.add_instance((f, pseudo_cls, d))

        if self.online_config["use_learned_stats"]:
            self.evaluation_online(sample_num, [[current_sample[0]], [current_sample[1]], [current_sample[2]]])

        if ((sample_num + 1) % self.online_config["update_interval"] != 0
                and not (sample_num == len(self.target_dataset))
                and self.online_config["update_interval"] >= sample_num):
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
            batch_size=self.online_config["batch_size"],
            shuffle=True,
            drop_last=False,
            pin_memory=False,
        )
        entropy_loss = HLoss(temp_factor=self.online_config["temp_factor"])

        for epoch in range(self.online_config["epochs"]):
            for feats in data_loader:
                feats = feats[0].to(self.device)
                preds = self.net(feats)

                if self.online_config["optimize"]:
                    loss = entropy_loss(preds)

                    self.online_optimizer.zero_grad()
                    loss.backward()
                    self.online_optimizer.step()

        return TRAINED

    def evaluation_online(self, epoch, current_sample):
        self.net.eval()

        with torch.no_grad():
            feats, cls, do = current_sample

            feats, cls, do = torch.stack(feats), torch.stack(cls), torch.stack(do)
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

    def dump_eval_online_result(self, is_train_offline=False):
        if is_train_offline:
            data_loader = torch.utils.data.DataLoader(self.target_dataset,
                                                      batch_size=self.train_config["batch_size"],
                                                      shuffle=False)

            for i, data in enumerate(data_loader):
                feats, cls, do = data
                input_data = [list(feats), list(cls), list(do)]
                self.evaluation_online(i * self.train_config["batch_size"], input_data)

        # logging json files
        json_file = open(self.save_path + 'online_eval.json', 'w')
        json_subsample = {key: self.json[key] for key in self.json.keys() - {'extracted_feat'}}
        json_file.write(to_json(json_subsample))
        json_file.close()

    def save_checkpoint(self, epoch):
        ckpt = {'model': self.net.state_dict(),
                'epoch': epoch,}
        torch.save(ckpt, self.save_path + "pretrained_checkpoint.pth")

    def load_checkpoint(self):
        if os.path.isfile(self.save_path + "pretrained_checkpoint.pth"):
            ckpt = torch.load(self.save_path + "pretrained_checkpoint.pth")
            self.net.load_state_dict(ckpt['model'])
            return ckpt['epoch']
        return 0

if __name__ == "__main__":
    model = NOTE()