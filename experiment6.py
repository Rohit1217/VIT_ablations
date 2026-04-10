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

def make_vit(patch_overlap):
    # patch_overlap=False → stride = patch_size (non-overlapping, 64 tokens)
    # patch_overlap=True  → stride = patch_size//2 (overlapping, 225 tokens)
    vit_cifar=VIT(image_size=cfg.VIT_IMAGE_SIZE,patch_size=cfg.VIT_PATCH_SIZE,
                  num_classes=cfg.VIT_NUM_CLASSES,num_blocks=cfg.VIT_NUM_BLOCKS,embed_dim=cfg.VIT_EMBED_DIM,
                  n_heads=cfg.VIT_N_HEADS,hidden_dim=cfg.VIT_HIDDEN_DIM,
                  max_seq_len=cfg.VIT_MAX_SEQ_LEN,patch_overlap=patch_overlap).to(Device)

    vit_cifar=torch.compile(vit_cifar)
    return vit_cifar





RESULTS_DIR = "results/experiment6"

# Experiment 6: Overlapping vs non-overlapping patch projection.
# Measures accuracy, training time, and peak GPU memory for each stride setting.
def run(patch_overlap_list=[False,True]):

    os.makedirs(RESULTS_DIR, exist_ok=True)

    vit_acc_dict={}
    vit_loss_dict={}
    vit_time_dict={}
    vit_mem_dict={}

    for patch_overlap in patch_overlap_list:
        label = "Overlap" if patch_overlap else "No Overlap"

        print(f"\n{'='*60}")
        print(f"Patch Overlap: {label}")
        print(f"{'='*60}")

        dataloaders=load_dataloaders(batch_size=batch_size,num_workers=num_workers)

        device_idx=int(Device.split(":")[-1])
        torch.cuda.set_device(device_idx)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        s_time=time.time()

        vit_cifar=make_vit(patch_overlap)

        vit_acc,vit_loss_list = train_model(vit_cifar,dataloaders,num_epochs,model_type=f"VIT_{label.replace(' ','_')}",Device=Device)

        torch.cuda.synchronize()
        elapsed=time.time()-s_time
        peak_mem_mb=torch.cuda.max_memory_allocated()/1024**2

        vit_loss_dict[patch_overlap]=vit_loss_list
        vit_acc_dict[patch_overlap]=vit_acc
        vit_time_dict[patch_overlap]=elapsed
        vit_mem_dict[patch_overlap]=peak_mem_mb

        print(f"ViT  best acc: {vit_acc:.4f} | peak mem: {peak_mem_mb:.1f} MB")

    # ── Plot: training loss curves ─────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    for patch_overlap in patch_overlap_list:
        label = "Overlap" if patch_overlap else "No Overlap"
        plt.plot(vit_loss_dict[patch_overlap], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Epoch — ViT Patch Overlap Ablation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = os.path.join(RESULTS_DIR, "experiment6_loss_curves.png")
    plt.savefig(loss_plot_path, dpi=150)
    plt.show()
    print(f"\nLoss plot saved to {loss_plot_path}")

    # ── Log results to file ────────────────────────────────────────────────────
    log_path = os.path.join(RESULTS_DIR, "results.txt")
    with open(log_path, "w") as f:
        f.write(f"{'Overlap':>12} | {'Best Val Acc':>14} | {'Train Time (s)':>16} | {'Peak Mem (MB)':>14}\n")
        f.write("-" * 66 + "\n")
        for patch_overlap in patch_overlap_list:
            label = "Overlap" if patch_overlap else "No Overlap"
            f.write(f"{label:>12} | {vit_acc_dict[patch_overlap]:>14.4f} | {vit_time_dict[patch_overlap]:>16.1f} | {vit_mem_dict[patch_overlap]:>14.1f}\n")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'Overlap':>12} | {'Best Val Acc':>14} | {'Train Time (s)':>16} | {'Peak Mem (MB)':>14}")
    print("-" * 66)
    for patch_overlap in patch_overlap_list:
        label = "Overlap" if patch_overlap else "No Overlap"
        print(f"{label:>12} | {vit_acc_dict[patch_overlap]:>14.4f} | {vit_time_dict[patch_overlap]:>16.1f} | {vit_mem_dict[patch_overlap]:>14.1f}")
    print(f"\nResults logged to {log_path}")



if __name__=="__main__":
    run()