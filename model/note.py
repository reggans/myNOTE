import torch
import torch.nn as nn
import torch.nn.functional as F

from

class NOTE:
    def __init__(self, device, source_dataloader, target_dataloader, save_path):
        self.device = device
        self.source_dataloader = source_dataloader
        self.target_dataloader = target_dataloader

        self.net =