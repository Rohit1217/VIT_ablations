from models.vit import VIT
from models.data import load_dataloaders

import torch
import torch.nn.functional as F
import config.config as cfg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
from tqdm import tqdm


os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_inductor_cache"
torch.set_float32_matmul_precision('high')

Device      = "cuda:0"
batch_size  = 10000
num_workers = cfg.NUM_WORKERS

RESULTS_DIR = "results/experiment5"
os.makedirs(RESULTS_DIR, exist_ok=True)

CIFAR_MEAN   = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
CIFAR_STD    = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
CIFAR_CLASSES = ['airplane','automobile','bird','cat','deer',
                 'dog','frog','horse','ship','truck']

grid_size  = cfg.VIT_IMAGE_SIZE // cfg.VIT_PATCH_SIZE 
num_layers = cfg.VIT_NUM_BLOCKS                        
n_heads    = cfg.VIT_N_HEADS                           
N_VIS      = 7                                          


def make_vit():
    vit = VIT(image_size=cfg.VIT_IMAGE_SIZE, patch_size=cfg.VIT_PATCH_SIZE,
              num_classes=cfg.VIT_NUM_CLASSES, num_blocks=cfg.VIT_NUM_BLOCKS,
              embed_dim=cfg.VIT_EMBED_DIM, n_heads=cfg.VIT_N_HEADS,
              hidden_dim=cfg.VIT_HIDDEN_DIM, max_seq_len=cfg.VIT_MAX_SEQ_LEN).to(Device)
    return torch.compile(vit)


# Load checkpoint
vit_cifar = make_vit()
ckpt = torch.load("best_VIT.pt")
vit_cifar.load_state_dict(ckpt)
vit_cifar = vit_cifar._orig_mod
vit_cifar.to(Device)
vit_cifar.eval()

_, test_loader = load_dataloaders(batch_size=batch_size, num_workers=num_workers)


def get_raw_attention(x):
    vit_cifar(x)
    atts = [block.att.att_weights.unsqueeze(1) for block in vit_cifar.transformer_block_list]
    return torch.cat(atts, dim=1)


def get_entropy_attention(x):
    vit_cifar(x)
    atts = []
    for block in vit_cifar.transformer_block_list:
        att = block.att.att_weights                      # (B, heads, seq, seq)
        att = -torch.log(att + 1e-16) * att
        att = torch.sum(att, dim=-1).unsqueeze(1)        # (B, 1, heads, seq)
        atts.append(att)
    return torch.cat(atts, dim=1)                        # (B, layers, heads, seq)


def denorm(img_t):
    """Denormalize (3,H,W) tensor → (H,W,3) numpy for imshow."""
    return (img_t.cpu() * CIFAR_STD + CIFAR_MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


x_all, y_all = next(iter(test_loader))
x_vis = x_all[:N_VIS].to(Device)
y_vis = y_all[:N_VIS]

with torch.no_grad():
    raw_att = get_raw_attention(x_vis)
    preds   = vit_cifar(x_vis).argmax(dim=1).cpu()


cls_att      = raw_att[:, :, :, -1, :-1].cpu()   
cls_att_mean = cls_att.mean(dim=2)                
# Reshape to spatial grid and upsample to full image resolution
cls_spatial = cls_att_mean.view(N_VIS, num_layers, grid_size, grid_size)
cls_up = F.interpolate(
    cls_spatial.view(N_VIS * num_layers, 1, grid_size, grid_size).float(),
    size=(cfg.VIT_IMAGE_SIZE, cfg.VIT_IMAGE_SIZE),
    mode='bilinear', align_corners=False
).view(N_VIS, num_layers, cfg.VIT_IMAGE_SIZE, cfg.VIT_IMAGE_SIZE)

# Normalize each map to [0,1] for consistent colour scaling
for i in range(N_VIS):
    for l in range(num_layers):
        m = cls_up[i, l]
        cls_up[i, l] = (m - m.min()) / (m.max() - m.min() + 1e-8)

# ── Per-image figures: original + 6 layer overlays ────────────────────────────
for i in range(N_VIS):
    fig, axes = plt.subplots(1, num_layers + 1, figsize=(14, 2.6))
    orig = denorm(x_vis[i])
    true_cls = CIFAR_CLASSES[y_vis[i]]
    pred_cls = CIFAR_CLASSES[preds[i]]

    axes[0].imshow(orig)
    axes[0].set_title(f"Input\ntrue: {true_cls}\npred: {pred_cls}", fontsize=7)
    axes[0].axis('off')

    for l in range(num_layers):
        axes[l + 1].imshow(orig)
        axes[l + 1].imshow(cls_up[i, l].numpy(), cmap='jet', alpha=0.5)
        axes[l + 1].set_title(f"Layer {l+1}", fontsize=8)
        axes[l + 1].axis('off')

    fig.suptitle(f"CLS Attention — Image {i+1} ({true_cls})", fontsize=9)
    fig.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f"attention_map_img{i+1}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")

# ── Combined grid: all N_VIS images × all layers ─────────────────────────────
fig, axes = plt.subplots(N_VIS, num_layers + 1, figsize=(14, N_VIS * 2.2))
for i in range(N_VIS):
    orig = denorm(x_vis[i])
    axes[i, 0].imshow(orig)
    axes[i, 0].set_ylabel(
        f"{CIFAR_CLASSES[y_vis[i]]}\n(pred: {CIFAR_CLASSES[preds[i]]})", fontsize=7
    )
    axes[i, 0].set_xticks([])
    axes[i, 0].set_yticks([])
    if i == 0:
        axes[i, 0].set_title("Input", fontsize=8)
    for l in range(num_layers):
        axes[i, l + 1].imshow(orig)
        axes[i, l + 1].imshow(cls_up[i, l].numpy(), cmap='jet', alpha=0.5)
        axes[i, l + 1].axis('off')
        if i == 0:
            axes[i, l + 1].set_title(f"L{l+1}", fontsize=8)

fig.suptitle("CLS Attention Maps across Layers (mean over heads)", fontsize=11)
fig.tight_layout()
grid_path = os.path.join(RESULTS_DIR, "attention_maps_grid.png")
fig.savefig(grid_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {grid_path}")

# ── Entropy over full test set ─────────────────────────────────────────────────
all_attentions = None
with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Entropy"):
        x = x.to(Device)
        att = get_entropy_attention(x)
        all_attentions = att if all_attentions is None else torch.cat([all_attentions, att], dim=0)

layer_entropy = torch.mean(all_attentions, dim=(0, 2, 3))   # (layers,)
head_entropy  = torch.mean(all_attentions, dim=(0, 3))       # (layers, heads)

torch.save(all_attentions.cpu(), os.path.join(RESULTS_DIR, "all_attentions.pt"))

with open(os.path.join(RESULTS_DIR, "results.txt"), "w") as f:
    f.write("Per-layer mean entropy\n")
    f.write("-" * 40 + "\n")
    for i, v in enumerate(layer_entropy.tolist()):
        f.write(f"  Layer {i+1}: {v:.4f}\n")
    f.write("\nPer-layer per-head mean entropy\n")
    f.write("-" * 40 + "\n")
    for i, row in enumerate(head_entropy.tolist()):
        vals = "  ".join(f"{v:.4f}" for v in row)
        f.write(f"  Layer {i+1}: [{vals}]\n")

print(f"\nAll results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    pass
