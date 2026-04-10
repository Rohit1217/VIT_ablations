from experiment1 import make_vit
from models.vit import VIT,Patchify
from models.models import *
from models.data import load_dataloaders
from experiment1 import *

import torch
import config.config as cfg

import os
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt

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


class VIT(nn.Module):
    def __init__(self,
                 image_size,
                 patch_size,
                 num_classes,
                 num_blocks,
                 embed_dim,
                 n_heads,
                 hidden_dim,
                 max_seq_len=1024,
                 use_cls=True,
                 pos_embed_type="Learn",
                 patch_overlap=False,
                 cls_pos="append"):
        super().__init__()

        self.transformer_block_list=nn.ModuleList([Transformer_block(embed_dim,
                                                                n_heads,
                                                                hidden_dim,
                                                                max_seq_len) for i in range(num_blocks)])

        if patch_overlap:
            out = (image_size - patch_size) // (patch_size // 2) + 1
            self.num_patches = out ** 2
        else:
            self.num_patches = (image_size // patch_size) ** 2

        if pos_embed_type =="Learn":
            self.pos_embed=nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        elif pos_embed_type=="Sin":
            pos_embed=Position_embedding(max_seq_len,embed_dim)
            self.register_buffer("pos_embed",pos_embed)
        else:
            pos_embed=torch.zeros(1, self.num_patches + 1, embed_dim)
            self.register_buffer("pos_embed",pos_embed)


        self.pos_embed_type=pos_embed_type

        self.patchifier=Patchify(image_size,patch_size,embed_dim,patch_overlap=patch_overlap)

        self.use_cls=use_cls
        self.cls_pos=cls_pos
        self.cls_token=nn.Parameter(torch.randn(1, 1, embed_dim))
        self.cls=nn.Linear(embed_dim,num_classes)
        self.ln_final=nn.LayerNorm(embed_dim)

    def forward(self,x):
        x=self.patchifier(x)
        B,C,D=x.shape
        layer_out_list=[]

        cls_token=self.cls_token.expand(x.shape[0],-1,-1)
        if self.cls_pos=="prepend":
            x=torch.concat([cls_token,x],dim=1)
        else:
            x=torch.concat([x,cls_token],dim=1)

        if self.pos_embed_type=="Sin":
            x+=self.pos_embed[None,:C+1,:]
        else:
            x+=self.pos_embed

        # layer 0: pre-transformer (patch embed + pos embed, before any transformer block)
        layer_out_list.append(x)

        for transformer_block in self.transformer_block_list:
            x=transformer_block(x)
            layer_out_list.append(x)

        if not self.use_cls:
            x_cls=self.cls(self.ln_final(x[:, 1:-1, :].mean(dim=1) if self.cls_pos=="prepend" else x[:, :-1, :].mean(dim=1)))
        else:
            x_cls=self.cls(self.ln_final(x[:, 0, :] if self.cls_pos=="prepend" else x[:, -1, :]))
        return x_cls,layer_out_list


class Linear_Probe(nn.Module):
    def __init__(self,in_dim,num_classes=10):
        super().__init__()
        self.fc1=nn.Linear(in_dim,num_classes)

    def forward(self,x):
        return self.fc1(x)



def make_vit():
    vit_cifar=VIT(image_size=cfg.VIT_IMAGE_SIZE,patch_size=cfg.VIT_PATCH_SIZE,
                  num_classes=cfg.VIT_NUM_CLASSES,num_blocks=cfg.VIT_NUM_BLOCKS,embed_dim=cfg.VIT_EMBED_DIM,
                  n_heads=cfg.VIT_N_HEADS,hidden_dim=cfg.VIT_HIDDEN_DIM,
                  max_seq_len=cfg.VIT_MAX_SEQ_LEN).to(Device)

    vit_cifar=torch.compile(vit_cifar)
    return vit_cifar


vit_cifar=make_vit()
ckpt=torch.load("best_VIT.pt")
vit_cifar.load_state_dict(ckpt)

vit_cifar=vit_cifar._orig_mod
vit_cifar.to(Device)


dataloaders=load_dataloaders(batch_size=batch_size,num_workers=num_workers)

vit_cifar.eval()



def train_probe(vit_model,linear_probe,idx,dataloaders,num_epochs,model_type="VIT",Device=Device):

    linear_probe_optim,linear_probe_scheduler=give_optim_scheduler(model=linear_probe,lr=lr,weight_decay=weight_decay,warmup_epochs=2)

    criterion=nn.CrossEntropyLoss(label_smoothing=0.1)

    train_loader,test_loader=dataloaders

    loss_list=[]
    best_acc=0.0

    vit_model.eval()
    for epoch in tqdm(range(num_epochs)):
        count=0
        train_loss=0

        linear_probe.train()

        train_correct = 0
        train_total   = 0

        for x,y in train_loader:
            x,y=x.to(Device),y.to(Device)
            with torch.no_grad():
                x=vit_model(x)[1][idx][:,-1,:].squeeze()
            y = y.long()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                y_pred=linear_probe(x)
                loss=criterion(y_pred, y)

            # fix: y is a class-index tensor (no mixup), not one-hot
            train_correct+=(y_pred.argmax(1) == y).sum().item()
            train_total+=y.size(0)
            loss.backward()

            linear_probe_optim.step()
            linear_probe_optim.zero_grad()

            train_loss+=loss.item()
            count+=1

        linear_probe_scheduler.step()
        linear_probe.eval()

        correct=0
        total=0

        train_acc=train_correct/train_total

        with torch.no_grad():
            for x, y in test_loader:
                x,y=x.to(Device), y.to(Device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # fix: extract VIT features before passing to probe
                    x_feat=vit_model(x)[1][idx][:,-1,:].squeeze()
                    y_pred=linear_probe(x_feat)
                correct+=(y_pred.argmax(1) == y).sum().item()
                total+=y.size(0)

        acc=correct/total
        print(f"epoch {epoch+1} | loss {train_loss/len(train_loader):.4f} | train_acc {train_acc:.4f} | val_acc {acc:.4f}")
        loss_list.append(train_loss/len(train_loader))

        if acc > best_acc:
            best_acc = acc
            print(f"  saved checkpoint — best acc {best_acc:.4f}")

    return best_acc,loss_list


RESULTS_DIR = "results/experiment7"

# layer 0 = pre-transformer embedding, layers 1..VIT_NUM_BLOCKS = transformer block outputs
NUM_LAYERS = cfg.VIT_NUM_BLOCKS + 1  # 7 total

def run():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    layer_accs  = []
    layer_losses = []

    for idx in range(NUM_LAYERS):
        layer_label = "embed" if idx == 0 else f"layer{idx}"
        print(f"\n{'='*60}")
        print(f"Linear probe at {layer_label} (idx={idx})")
        print(f"{'='*60}")

        linear_probe = Linear_Probe(cfg.VIT_EMBED_DIM, cfg.VIT_NUM_CLASSES).to(Device)
        best_acc, loss_list = train_probe(
            vit_cifar, linear_probe, idx, dataloaders, num_epochs
        )

        layer_accs.append(best_acc)
        layer_losses.append(loss_list)

        print(f"{layer_label} best acc: {best_acc:.4f}")

    # ── Plot 1: accuracy vs layer ──────────────────────────────────────────────
    layer_labels = ["embed"] + [f"L{i}" for i in range(1, NUM_LAYERS)]

    plt.figure(figsize=(9, 5))
    plt.plot(layer_labels, layer_accs, marker='o')
    plt.xlabel("Layer")
    plt.ylabel("Test Accuracy")
    plt.title("Linear Probe Accuracy by ViT Layer — CIFAR-10")
    plt.grid(True)
    plt.tight_layout()
    acc_plot_path = os.path.join(RESULTS_DIR, "experiment7_accuracy_vs_layer.png")
    plt.savefig(acc_plot_path, dpi=150)
    plt.close()
    print(f"\nAccuracy plot saved to {acc_plot_path}")

    # ── Plot 2: loss curves per layer ─────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    for label, losses in zip(layer_labels, layer_losses):
        plt.plot(losses, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Linear Probe Loss Curves by Layer")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = os.path.join(RESULTS_DIR, "experiment7_loss_curves.png")
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    print(f"Loss plot saved to {loss_plot_path}")

    # ── Log results to file ────────────────────────────────────────────────────
    log_path = os.path.join(RESULTS_DIR, "results.txt")
    with open(log_path, "w") as f:
        f.write(f"{'Layer':>10} | {'Val Acc':>10}\n")
        f.write("-" * 25 + "\n")
        for label, acc in zip(layer_labels, layer_accs):
            f.write(f"{label:>10} | {acc:>10.4f}\n")

    print(f"\n{'Layer':>10} | {'Val Acc':>10}")
    print("-" * 25)
    for label, acc in zip(layer_labels, layer_accs):
        print(f"{label:>10} | {acc:>10.4f}")
    print(f"\nResults logged to {log_path}")


if __name__=="__main__":
    run()