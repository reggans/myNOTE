import argparse
import os

import numpy as np
import torch

from model.note import NOTE
from dataloader.CIFAR10Dataset import CIFAR10Dataset
from config import NOTE_CONFIG, CIFAR10_CONFIG

corruptions = ["shot_noise-5", "motion_blur-5", "snow-5", "pixelate-5", "gaussian_noise-5", "defocus_blur-5",
                    "brightness-5", "fog-5", "zoom_blur-5", "frost-5", "glass_blur-5", "impulse_noise-5", "contrast-5",
                    "jpeg_compression-5", "elastic_transform-5"]

if __name__ == "__main__":
    NOTE_CONFIG["train_config"]["lr"] = CIFAR10_CONFIG["learning_rate"]
    NOTE_CONFIG["train_config"]["batch_size"] = CIFAR10_CONFIG["batch_size"]
    NOTE_CONFIG["train_config"]["momentum"] = CIFAR10_CONFIG["momentum"]
    NOTE_CONFIG["train_config"]["weight_decay"] = CIFAR10_CONFIG["weight_decay"]
    NOTE_CONFIG["train_config"]["epochs"] = CIFAR10_CONFIG["epochs"]

    source_dataset = CIFAR10Dataset(
        file_path=CIFAR10_CONFIG["file_path"],
        domains=["original"],
        transform='src',
    )

    target_dataset = CIFAR10Dataset(
        file_path=CIFAR10_CONFIG["file_path"],
        domains=["shot_noise-5"],
        transform='tgt',
        distribution=CIFAR10_CONFIG["distribution"],
        dir_beta=CIFAR10_CONFIG["dir_beta"],
    )

    model = NOTE(source_dataset, target_dataset, **NOTE_CONFIG)
    start_epoch = model.load_checkpoint()

    best_acc, best_epoch = 0, 0
    for epoch in range(start_epoch, NOTE_CONFIG["train_config"]["epochs"]):
        model.train()
        avg_loss, avg_acc = model.evaluation()

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_epoch = epoch

        model.save_checkpoint(epoch)

    print(f'best_acc: {best_acc}, best_epoch: {best_epoch}')

    for epoch in range(len(target_dataset)):
        model.train_online(epoch, adapt=True)

    model.dump_eval_online_result()