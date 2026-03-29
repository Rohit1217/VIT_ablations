from experiment1 import make_vit,give_transforms,give_optim_scheduler,train_model
from models.vit import VIT
from models.data import load_dataloaders

import torch,torchvision
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR,CosineAnnealingLR,SequentialLR
from torch.optim import AdamW
import torchvision.transforms.v2 as tf
import config.config as cfg
import time

import os
from tqdm import tqdm
import matplotlib.pyplot as plt

print(torchvision.__version__)

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_inductor_cache"
torch.set_float32_matmul_precision('high')


Device="cuda:2"
num_epochs=cfg.NUM_EPOCHS
warmup_epochs=cfg.WARMUP_EPOCHS
weight_decay=cfg.WEIGHT_DECAY
lr=cfg.LR

batch_size=cfg.BATCH_SIZE
num_workers=cfg.NUM_WORKERS
data_frac=cfg.DATA_FRAC

def make_vit(use_cls):
    vit_cifar=VIT(image_size=cfg.VIT_IMAGE_SIZE,patch_size=cfg.VIT_PATCH_SIZE,
                  num_classes=cfg.VIT_NUM_CLASSES,num_blocks=cfg.VIT_NUM_BLOCKS,embed_dim=cfg.VIT_EMBED_DIM,
                  n_heads=cfg.VIT_N_HEADS,hidden_dim=cfg.VIT_HIDDEN_DIM,
                  max_seq_len=cfg.VIT_MAX_SEQ_LEN,use_cls=use_cls).to(Device)

    vit_cifar=torch.compile(vit_cifar)
    return vit_cifar





RESULTS_DIR = "results/experiment3"

def run(use_cls_list=[True,False]):

    os.makedirs(RESULTS_DIR, exist_ok=True)

    vit_acc_dict={}
    vit_loss_dict={}
    vit_time_dict={}

    for use_cls in use_cls_list:
        s_time=time.time()
        label = "CLS token" if use_cls else "Mean pool"

        print(f"\n{'='*60}")
        print(f"Pooling: {label}")
        print(f"{'='*60}")

        dataloaders=load_dataloaders(batch_size=batch_size,num_workers=num_workers)

        vit_cifar=make_vit(use_cls)

        vit_acc,vit_loss_list = train_model(vit_cifar,dataloaders,num_epochs,model_type=f"VIT_{label.replace(' ','_')}",Device=Device)

        vit_loss_dict[use_cls]=vit_loss_list
        vit_acc_dict[use_cls]=vit_acc
        vit_time_dict[use_cls]=time.time()-s_time

        print(f"ViT  best acc: {vit_acc:.4f}")

    # ── Plot: training loss curves per pooling method ──────────────────────────
    plt.figure(figsize=(8, 5))
    for use_cls in use_cls_list:
        label = "CLS token" if use_cls else "Mean pool"
        plt.plot(vit_loss_dict[use_cls], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Epoch — ViT CLS Token vs Mean Pooling")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = os.path.join(RESULTS_DIR, "experiment3_loss_curves.png")
    plt.savefig(loss_plot_path, dpi=150)
    plt.show()
    print(f"\nLoss plot saved to {loss_plot_path}")

    # ── Log results to file ────────────────────────────────────────────────────
    log_path = os.path.join(RESULTS_DIR, "results.txt")
    with open(log_path, "w") as f:
        f.write(f"{'Pooling':>12} | {'Best Val Acc':>14} | {'Train Time (s)':>16}\n")
        f.write("-" * 48 + "\n")
        for use_cls in use_cls_list:
            label = "CLS token" if use_cls else "Mean pool"
            f.write(f"{label:>12} | {vit_acc_dict[use_cls]:>14.4f} | {vit_time_dict[use_cls]:>16.1f}\n")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'Pooling':>12} | {'Best Val Acc':>14} | {'Train Time (s)':>16}")
    print("-" * 48)
    for use_cls in use_cls_list:
        label = "CLS token" if use_cls else "Mean pool"
        print(f"{label:>12} | {vit_acc_dict[use_cls]:>14.4f} | {vit_time_dict[use_cls]:>16.1f}")
    print(f"\nResults logged to {log_path}")



if __name__=="__main__":
    run()