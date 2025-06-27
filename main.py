import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import wandb

from model.note import NOTE
from dataloader.CIFAR10Dataset import CIFAR10Dataset
from config import CIFAR10_CONFIG, PRETRAIN_CONFIG, ONLINE_CONFIG

corruptions = ["shot_noise-5", "motion_blur-5", "snow-5", "pixelate-5", "gaussian_noise-5", "defocus_blur-5",
                    "brightness-5", "fog-5", "zoom_blur-5", "frost-5", "glass_blur-5", "impulse_noise-5", "contrast-5",
                    "jpeg_compression-5", "elastic_transform-5"]

if __name__ == "__main__":
    source_dataset = CIFAR10Dataset(
        file_path=CIFAR10_CONFIG["file_path"],
        domains=["original"],
        transform='src',
    )

    val_dataset = CIFAR10Dataset(
        file_path=CIFAR10_CONFIG["file_path"],
        domains=["test"],
        transform='tgt',
        distribution='random'
    )

    target_dataset = CIFAR10Dataset(
        file_path=CIFAR10_CONFIG["file_path"],
        domains=["shot_noise-5"],
        transform='tgt',
        distribution=CIFAR10_CONFIG["distribution"],
        dir_beta=CIFAR10_CONFIG["dir_beta"],
    )

    model = NOTE(source_dataset, val_dataset, target_dataset,
                 save_path="logs/NOTE/", checkpoint_path="ckpt/NOTE/",
                 **PRETRAIN_CONFIG)
    start_epoch = model.load_checkpoint()
    # start_epoch = 0
    print(f"Starting epoch {start_epoch}")

    wandb.login()
    run = wandb.init(
        project="myNOTE",
        config=PRETRAIN_CONFIG,
    )

    best_acc, best_epoch = 0, 0
    for epoch in range(start_epoch, model.epochs):
        model.train()
        avg_loss, avg_acc = model.evaluation()

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_epoch = epoch

        wandb.log({"accuracy": avg_acc, "loss": avg_loss})

        model.save_checkpoint(epoch)

    # Loaded fully-trained
    if best_acc == 0:
        _, best_acc = model.evaluation()
        best_epoch = model.epochs

    print(f'best_acc: {best_acc}, best_epoch: {best_epoch}')

    # Online
    model.setup_online(ONLINE_CONFIG)
    for epoch in range(len(target_dataset)):
        model.train_online(epoch, adapt=True)

    model.dump_eval_online_result()