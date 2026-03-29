"""
Experiment 1: Accuracy vs Training Data Size
Trains ViT and ResNet on CIFAR-10 at 5%, 10%, 25%, 50%, 100% of training data.
Plots test accuracy vs data fraction for both models.
"""

import torch
import torch.nn as nn
import torchvision.transforms.v2 as tf
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import load_dataloaders
from vit import VIT
from resnet import Resnet
import resnet_config as rcfg
import os


os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_inductor_cache"
torch.set_float32_matmul_precision('high')

# ── Experiment config ──────────────────────────────────────────────────────────
FRACTIONS   = [0.05, 0.10, 0.25, 0.50, 1.00]
DEVICE      = "cuda:2"
NUM_EPOCHS  = 200          # reduced from 200 for experiment speed
BATCH_SIZE  = 512
NUM_WORKERS = 16

CIFAR_MEAN  = (0.4914, 0.4822, 0.4465)
CIFAR_STD   = (0.2470, 0.2435, 0.2616)

# ── Model factory ──────────────────────────────────────────────────────────────
def make_vit():
    return VIT(image_size=32, patch_size=4, num_classes=10,
               num_blocks=6, embed_dim=256, n_heads=8,
               hidden_dim=1024, max_seq_len=100)

def make_resnet():
    return Resnet(num_layers=rcfg.NUM_LAYERS,
                  proj_kernel=rcfg.PROJ_KERNEL,
                  residual_channels=rcfg.RESIDUAL_CHANNELS,
                  residual_kernel=rcfg.RESIDUAL_KERNEL,
                  stride=rcfg.STRIDE,
                  padding=rcfg.PADDING,
                  bias=rcfg.BIAS,
                  batch_norm=rcfg.BATCH_NORM,
                  hidden_dim=rcfg.HIDDEN_DIM,
                  num_classes=10)

# ── Core training loop ─────────────────────────────────────────────────────────
def train_model(model, train_loader, test_loader, num_epochs, label):
    model = model.to(DEVICE)
    model = torch.compile(model)

    gpu_transforms = tf.Compose([
        tf.RandomCrop(32, padding=4),
        tf.RandomHorizontalFlip(),
        tf.ColorJitter(0.1, 0.1, 0.1),
        tf.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]).to(DEVICE)

    mixup     = tf.MixUp(alpha=0.2, num_classes=10)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    warmup    = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=10)
    cosine    = CosineAnnealingLR(optimizer, num_epochs - 10)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[10])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0

    for _ in tqdm(range(num_epochs), desc=label):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = gpu_transforms(x)
            y = y.long()
            x, y = mixup(x, y)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = criterion(model(x), y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    y_pred = model(x)
                correct += (y_pred.argmax(1) == y).sum().item()
                total   += y.size(0)

        acc = correct / total
        if acc > best_acc:
            best_acc = acc

    return best_acc


def get_subset_loader(full_train_dataset, fraction):
    n       = len(full_train_dataset)
    k       = int(n * fraction)
    indices = torch.randperm(n)[:k]
    subset  = Subset(full_train_dataset, indices)
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=NUM_WORKERS, pin_memory=True,
                      prefetch_factor=2, persistent_workers=True)


# ── Experiment ─────────────────────────────────────────────────────────────────
def run():
    torch.set_float32_matmul_precision('high')

    full_train_loader, test_loader = load_dataloaders(BATCH_SIZE, num_workers=NUM_WORKERS)
    full_train_dataset = full_train_loader.dataset

    vit_accs    = []
    resnet_accs = []

    for frac in FRACTIONS:
        print(f"\n{'='*60}")
        print(f"Fraction: {frac:.0%}  ({int(len(full_train_dataset)*frac):,} samples)")
        print(f"{'='*60}")

        train_loader = get_subset_loader(full_train_dataset, frac)

        vit_acc = train_model(make_vit(), train_loader, test_loader,
                              NUM_EPOCHS, label=f"ViT  {frac:.0%}")
        resnet_acc = train_model(make_resnet(), train_loader, test_loader,
                                 NUM_EPOCHS, label=f"ResNet {frac:.0%}")

        vit_accs.append(vit_acc)
        resnet_accs.append(resnet_acc)

        print(f"  ViT    best acc: {vit_acc:.4f}")
        print(f"  ResNet best acc: {resnet_acc:.4f}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    pct = [f * 100 for f in FRACTIONS]

    plt.figure(figsize=(8, 5))
    plt.plot(pct, vit_accs,    marker='o', label='ViT')
    plt.plot(pct, resnet_accs, marker='s', label='ResNet (CNN)')
    plt.xlabel("Training Data (%)")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy vs Training Data Size — ViT vs ResNet on CIFAR-10")
    plt.xticks(pct, [f"{p:.0f}%" for p in pct])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("experiment1_accuracy_vs_data.png", dpi=150)
    plt.show()
    print("\nPlot saved to experiment1_accuracy_vs_data.png")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'Fraction':>10} | {'ViT Acc':>10} | {'ResNet Acc':>10}")
    print("-" * 36)
    for frac, va, ra in zip(FRACTIONS, vit_accs, resnet_accs):
        print(f"{frac:>10.0%} | {va:>10.4f} | {ra:>10.4f}")


if __name__ == "__main__":
    run()
