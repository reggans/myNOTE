import argparse

import numpy as np
import torch

from model.note import NOTE
from dataloader.CIFAR10Dataset import CIFAR10Dataset
from config import NOTE_CONFIG, CIFAR10_CONFIG

corruptions = ["shot_noise-5", "motion_blur-5", "snow-5", "pixelate-5", "gaussian_noise-5", "defocus_blur-5",
                    "brightness-5", "fog-5", "zoom_blur-5", "frost-5", "glass_blur-5", "impulse_noise-5", "contrast-5",
                    "jpeg_compression-5", "elastic_transform-5"]

if __name__ == "__main__":
    source_dataset = CIFAR10Dataset(
        file_path=CIFAR10_CONFIG["file_path"],
        domains=["original"],
        transform='src',
    )
    print(len(source_dataset))

    target_dataset = CIFAR10Dataset(
        file_path=CIFAR10_CONFIG["file_path"],
        domains=["test"],
        transform='tgt',
        distribution=CIFAR10_CONFIG["distribution"],
        dir_beta=CIFAR10_CONFIG["dir_beta"],
    )
    print(len(target_dataset))
    exit()

    model = NOTE(source_dataset, target_dataset, **NOTE_CONFIG)

    best_acc, best_epoch = 0, 0
    for epoch in range(NOTE_CONFIG["train_config"]["epochs"]):
        model.train()
        avg_loss, avg_acc = model.evaluation()

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_epoch = epoch

    print(f'best_acc: {best_acc}, best_epoch: {best_epoch}')

    for epoch in range(len(target_dataset)):
        model.train_online(epoch, adapt=True)

    model.dump_eval_online_result()