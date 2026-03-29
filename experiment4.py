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


Device="cuda:0"
num_epochs=cfg.NUM_EPOCHS
warmup_epochs=cfg.WARMUP_EPOCHS
weight_decay=cfg.WEIGHT_DECAY
lr=cfg.LR

batch_size=cfg.BATCH_SIZE
num_workers=cfg.NUM_WORKERS
data_frac=cfg.DATA_FRAC

def make_vit(pos_embed_type):
    vit_cifar=VIT(image_size=cfg.VIT_IMAGE_SIZE,patch_size=cfg.VIT_PATCH_SIZE,
                  num_classes=cfg.VIT_NUM_CLASSES,num_blocks=cfg.VIT_NUM_BLOCKS,embed_dim=cfg.VIT_EMBED_DIM,
                  n_heads=cfg.VIT_N_HEADS,hidden_dim=cfg.VIT_HIDDEN_DIM,
                  max_seq_len=cfg.VIT_MAX_SEQ_LEN,pos_embed_type=pos_embed_type).to(Device)

    vit_cifar=torch.compile(vit_cifar)
    return vit_cifar





RESULTS_DIR = "results/experiment4"

def run(pos_embed_type_list=["Learn","Sin","None"]):

    os.makedirs(RESULTS_DIR, exist_ok=True)

    vit_acc_dict={}
    vit_loss_dict={}
    vit_time_dict={}

    for pos_embed_type in pos_embed_type_list:
        s_time=time.time()
        label = f"{pos_embed_type}" 

        print(f"\n{'='*60}")
        print(f"Position_embed: {label}")
        print(f"{'='*60}")

        dataloaders=load_dataloaders(batch_size=batch_size,num_workers=num_workers)

        vit_cifar=make_vit(pos_embed_type)

        vit_acc,vit_loss_list = train_model(vit_cifar,dataloaders,num_epochs,model_type=f"VIT_{label.replace(' ','_')}",Device=Device)

        vit_loss_dict[pos_embed_type]=vit_loss_list
        vit_acc_dict[pos_embed_type]=vit_acc
        vit_time_dict[pos_embed_type]=time.time()-s_time

        print(f"ViT  best acc: {vit_acc:.4f}")

    # ── Plot: training loss curves per pos embed type ──────────────────────────
    plt.figure(figsize=(8, 5))
    for pos_embed_type in pos_embed_type_list:
        plt.plot(vit_loss_dict[pos_embed_type], label=pos_embed_type)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Epoch — ViT Positional Embedding Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = os.path.join(RESULTS_DIR, "experiment4_loss_curves.png")
    plt.savefig(loss_plot_path, dpi=150)
    plt.show()
    print(f"\nLoss plot saved to {loss_plot_path}")

    # ── Log results to file ────────────────────────────────────────────────────
    log_path = os.path.join(RESULTS_DIR, "results.txt")
    with open(log_path, "w") as f:
        f.write(f"{'Pos Embed':>12} | {'Best Val Acc':>14} | {'Train Time (s)':>16}\n")
        f.write("-" * 48 + "\n")
        for pos_embed_type in pos_embed_type_list:
            f.write(f"{pos_embed_type:>12} | {vit_acc_dict[pos_embed_type]:>14.4f} | {vit_time_dict[pos_embed_type]:>16.1f}\n")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'Pos Embed':>12} | {'Best Val Acc':>14} | {'Train Time (s)':>16}")
    print("-" * 48)
    for pos_embed_type in pos_embed_type_list:
        print(f"{pos_embed_type:>12} | {vit_acc_dict[pos_embed_type]:>14.4f} | {vit_time_dict[pos_embed_type]:>16.1f}")
    print(f"\nResults logged to {log_path}")



if __name__=="__main__":
    run()