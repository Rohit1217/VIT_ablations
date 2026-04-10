
from models.data import load_dataloaders
from models.vit import VIT
from models.resnet import Resnet

import torch,torchvision
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR,CosineAnnealingLR,SequentialLR
from torch.optim import AdamW
import torchvision.transforms.v2 as tf
import config.config as cfg

import os
from tqdm import tqdm
import matplotlib.pyplot as plt

print(torchvision.__version__)

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_inductor_cache"
torch.set_float32_matmul_precision('high')


Device=cfg.DEVICE
num_epochs=cfg.NUM_EPOCHS
warmup_epochs=cfg.WARMUP_EPOCHS
weight_decay=cfg.WEIGHT_DECAY
lr=cfg.LR

batch_size=cfg.BATCH_SIZE
num_workers=cfg.NUM_WORKERS
data_frac=cfg.DATA_FRAC

def make_vit():
    # Build ViT with default hyperparams from config
    vit_cifar=VIT(image_size=cfg.VIT_IMAGE_SIZE,patch_size=cfg.VIT_PATCH_SIZE,
                  num_classes=cfg.VIT_NUM_CLASSES,num_blocks=cfg.VIT_NUM_BLOCKS,embed_dim=cfg.VIT_EMBED_DIM,
                  n_heads=cfg.VIT_N_HEADS,hidden_dim=cfg.VIT_HIDDEN_DIM,
                  max_seq_len=cfg.VIT_MAX_SEQ_LEN).to(cfg.DEVICE)
    
    vit_cifar=torch.compile(vit_cifar)
    return vit_cifar

def make_resnet():
    # Build ResNet with default hyperparams from config
    resnet_cifar=Resnet(num_layers=cfg.RES_NUM_LAYERS,
                  proj_kernel=cfg.RES_PROJ_KERNEL,
                  residual_channels=cfg.RES_RESIDUAL_CHANNELS,
                  residual_kernel=cfg.RES_RESIDUAL_KERNEL,
                  stride=cfg.RES_STRIDE,
                  padding=cfg.RES_PADDING,
                  bias=cfg.RES_BIAS,
                  batch_norm=cfg.RES_BATCH_NORM,
                  hidden_dim=cfg.RES_HIDDEN_DIM,
                  num_classes=cfg.NUM_CLASSES).to(Device)
    
    resnet_cifar=torch.compile(resnet_cifar)
    return resnet_cifar


def give_transforms(Device=Device):
    gpu_transforms = tf.Compose([
        tf.RandomCrop(32, padding=4),
        tf.RandomHorizontalFlip(),
        tf.ColorJitter(0.1, 0.1, 0.1),
        tf.Normalize(cfg.CIFAR_MEAN, cfg.CIFAR_STD),
    ]).to(Device)

    mixup=tf.MixUp(alpha=0.2, num_classes=10)

    return gpu_transforms,mixup


def give_optim_scheduler(model,lr,weight_decay,warmup_epochs):
    model_optim=AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    warmup=LinearLR(model_optim,start_factor=0.01,end_factor=1,total_iters=warmup_epochs)
    cosine_decay=CosineAnnealingLR(model_optim,num_epochs-warmup_epochs)
    model_scheduler=SequentialLR(model_optim,[warmup,cosine_decay],milestones=[warmup_epochs])

    return model_optim,model_scheduler



def train_model(model,dataloaders,num_epochs,model_type="VIT",Device=Device):
    
    model_optim,model_scheduler=give_optim_scheduler(model=model,lr=lr,weight_decay=weight_decay,warmup_epochs=warmup_epochs)

    criterion=nn.CrossEntropyLoss(label_smoothing=0.1)

    train_loader,test_loader=dataloaders
    gpu_transforms,mixup=give_transforms(Device)

    loss_list=[]
    # prev_loss=torch.tensor(1e16)
    best_acc=0.0

    for epoch in tqdm(range(num_epochs)):
        count=0
        train_loss=0
        
        model.train()
        
        train_correct = 0
        train_total   = 0
        
        for x,y in train_loader:
            x,y=x.to(Device),y.to(Device)
            x=gpu_transforms(x)
            y = y.long()   
            x,y=mixup(x,y)
    
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                y_pred=model(x)
                loss=criterion(y_pred, y)

            train_correct+=(y_pred.argmax(1) == y.argmax(1)).sum().item()
            train_total+=y.size(0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            model_optim.step()
            model_optim.zero_grad()

            train_loss+=loss.item()
            count+=1
        
        model_scheduler.step()
        model.eval()
        
        correct=0
        total=0

        train_acc=train_correct/train_total

        with torch.no_grad():
            for x, y in test_loader:
                x,y=x.to(Device), y.to(Device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    y_pred=model(x)
                correct+=(y_pred.argmax(1) == y).sum().item()
                total+=y.size(0)

        acc=correct/total
        print(f"epoch {epoch+1} | loss {train_loss/len(train_loader):.4f} | train_acc {train_acc:.4f} | val_acc {acc:.4f}")
        loss_list.append(train_loss/len(train_loader))

        # ── checkpoint ─────────────────────────────────────
        if acc > best_acc:
            best_acc = acc
            # torch.save(model.state_dict(), f"best_{model_type}.pt")
            print(f"  saved checkpoint — best acc {best_acc:.4f}")
        
    return best_acc,loss_list



RESULTS_DIR = "results/experiment1"

# Experiment 1: Data efficiency — ViT vs ResNet.
# Trains both models at 5%, 10%, 25%, 50%, 100% of CIFAR-10 training data.
def run(data_frac_list=[0.05,0.1,0.25,0.5,1]):
    torch.set_float32_matmul_precision('high')

    os.makedirs(RESULTS_DIR, exist_ok=True)

    vit_accs      = []
    resnet_accs   = []
    vit_losses    = []
    resnet_losses = []

    for frac in data_frac_list:
        print(f"\n{'='*60}")
        print(f"Fraction: {frac:.0%}  ({int(50000*frac):,} samples)")
        print(f"{'='*60}")

        dataloaders=load_dataloaders(batch_size=batch_size,num_workers=num_workers,data_frac=frac)

        vit_cifar=make_vit()
        resnet_cifar=make_resnet()

        vit_acc,vit_loss_list = train_model(vit_cifar,dataloaders,num_epochs,model_type="VIT")
        resnet_acc,resnet_loss_list = train_model(resnet_cifar,dataloaders,num_epochs,model_type="RESNET")

        vit_accs.append(vit_acc)
        resnet_accs.append(resnet_acc)
        vit_losses.append(vit_loss_list)
        resnet_losses.append(resnet_loss_list)

        print(f"ViT    best acc: {vit_acc:.4f}")
        print(f"ResNet best acc: {resnet_acc:.4f}")

    # ── Plot 1: accuracy vs data fraction ─────────────────────────────────────
    pct = [f * 100 for f in data_frac_list]

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
    acc_plot_path = os.path.join(RESULTS_DIR, "experiment1_accuracy_vs_data.png")
    plt.savefig(acc_plot_path, dpi=150)
    plt.close()
    print(f"\nAccuracy plot saved to {acc_plot_path}")

    # ── Plot 2: loss curves per fraction ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, losses, title in zip(axes,
                                  [vit_losses, resnet_losses],
                                  ["ViT", "ResNet"]):
        for frac, loss in zip(data_frac_list, losses):
            ax.plot(loss, label=f"{frac:.0%}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss")
        ax.set_title(f"{title} Loss Curves")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    loss_plot_path = os.path.join(RESULTS_DIR, "experiment1_loss_curves.png")
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    print(f"Loss plot saved to {loss_plot_path}")

    # ── Log results to file ────────────────────────────────────────────────────
    log_path = os.path.join(RESULTS_DIR, "results.txt")
    with open(log_path, "w") as f:
        f.write(f"{'Fraction':>10} | {'ViT Acc':>10} | {'ResNet Acc':>10}\n")
        f.write("-" * 36 + "\n")
        for frac, va, ra in zip(data_frac_list, vit_accs, resnet_accs):
            f.write(f"{frac:>10.0%} | {va:>10.4f} | {ra:>10.4f}\n")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'Fraction':>10} | {'ViT Acc':>10} | {'ResNet Acc':>10}")
    print("-" * 36)
    for frac, va, ra in zip(data_frac_list, vit_accs, resnet_accs):
        print(f"{frac:>10.0%} | {va:>10.4f} | {ra:>10.4f}")
    print(f"\nResults logged to {log_path}")


if __name__ == "__main__":
    run()
