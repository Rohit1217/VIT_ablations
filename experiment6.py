from experiment1 import make_vit
from models.vit import VIT
from models.data import load_dataloaders

import torch
import config.config as cfg

import os
from tqdm import tqdm


os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_inductor_cache"
torch.set_float32_matmul_precision('high')


Device="cuda:0"


batch_size=10000
num_workers=cfg.NUM_WORKERS
data_frac=cfg.DATA_FRAC

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


train_loader,test_loader=load_dataloaders(batch_size=batch_size,num_workers=num_workers)

vit_cifar.eval()


def get_attention_outputs(x):
    all_attentions=None
    out=vit_cifar(x)
    for i,block in enumerate(vit_cifar.transformer_block_list):
        att=block.att.att_weights
        att=-torch.log(att+1e-16)*att
        att=torch.sum(att,dim=-1)
        att=att.unsqueeze(1)
        
        if all_attentions is None:
            all_attentions=att
        else:
            all_attentions=torch.concat([all_attentions,att],dim=1)

    return all_attentions


all_attentions=None
with torch.no_grad():
    for x, y in tqdm(test_loader):
        x,y=x.to(Device), y.to(Device)

        att=get_attention_outputs(x)
        if all_attentions is None:
            all_attentions=att
        else:
            all_attentions=torch.concat([all_attentions,att],dim=0)
        

layer_entropy = torch.mean(all_attentions, dim=(0,2,3))   # (6,)
head_entropy  = torch.mean(all_attentions, dim=(0,3))     # (6, 8)

print(all_attentions.shape, layer_entropy, head_entropy)

RESULTS_DIR = "results/experiment6"
os.makedirs(RESULTS_DIR, exist_ok=True)

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

print(f"Results saved to {RESULTS_DIR}")
exit()
