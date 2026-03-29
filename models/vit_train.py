
from data import load_dataloaders
from vit import VIT

import torch,torchvision
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR,CosineAnnealingLR,SequentialLR
from torch.optim import AdamW
import torchvision.transforms.v2 as tf

from tqdm import tqdm

print(torchvision.__version__)

torch.set_float32_matmul_precision('high')


image_size=32
patch_size=4      
num_classes=10
num_blocks=6
embed_dim=256
n_heads=8
hidden_dim=1024     
max_seq_len=100 

Device="cuda:1"


vit_cifar=VIT(image_size,
              patch_size,
              num_classes,
              num_blocks,
              embed_dim,
              n_heads,
              hidden_dim,
              max_seq_len)

trainable_params = sum(p.numel() for p in vit_cifar.parameters() if p.requires_grad)
print(f"Trainable params: {trainable_params:,}")
# print(1+"sd")

train_loader,test_loader=load_dataloaders(512,num_workers=16,data_frac=1)

gpu_transforms=tf.Compose([tf.RandomCrop(32, padding=4),
                    tf.RandomHorizontalFlip(),
                    tf.ColorJitter(0.1, 0.1, 0.1),
                        tf.Normalize((0.4914, 0.4822, 0.4465),
                 (0.2470, 0.2435, 0.2616))] ).to(Device)

mixup=tf.MixUp(alpha=0.2, num_classes=10)



num_epochs=200
warmup_epochs=10
tol=1e-3
lr=1e-3
weight_decay=0.1

label_smoothing=0.1


vit_cifar=vit_cifar.to(Device)
vit_cifar = torch.compile(vit_cifar)
vit_optim=AdamW(vit_cifar.parameters(),lr,weight_decay=weight_decay)

warmup=LinearLR(vit_optim,start_factor=0.01,end_factor=1,total_iters=warmup_epochs)
cosine_decay=CosineAnnealingLR(vit_optim,num_epochs-warmup_epochs)
scheduler=SequentialLR(vit_optim,[warmup,cosine_decay],milestones=[warmup_epochs])

criterion=nn.CrossEntropyLoss()

loss_list=[]
prev_loss=torch.tensor(1e16)
best_acc=0.0

for epoch in tqdm(range(num_epochs)):
    count=0
    train_loss=0
    vit_cifar.train()
    train_correct = 0
    train_total   = 0
    
    for x,y in train_loader:
        x,y=x.to(Device),y.to(Device)
        x=gpu_transforms(x)
        y = y.long()   
        x,y=mixup(x,y)
 
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y_pred=vit_cifar(x)
            loss=criterion(y_pred, y)

        train_correct+=(y_pred.argmax(1) == y.argmax(1)).sum().item()
        train_total+=y.size(0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vit_cifar.parameters(), max_norm=1.0)

        vit_optim.step()
        vit_optim.zero_grad()

        train_loss+=loss.item()
        count+=1
    
    scheduler.step()
    vit_cifar.eval()
    correct = 0
    total   = 0

    train_acc = train_correct / train_total

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(Device), y.to(Device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                y_pred=vit_cifar(x)
            correct+=(y_pred.argmax(1) == y).sum().item()
            total+=y.size(0)

    acc = correct / total
    print(f"epoch {epoch+1} | loss {train_loss/len(train_loader):.4f} | train_acc {train_acc:.4f} | val_acc {acc:.4f}")


    # ── checkpoint ─────────────────────────────────────
    if acc > best_acc:
        best_acc = acc
        torch.save(vit_cifar.state_dict(), "best_vit.pt")
        print(f"  saved checkpoint — best acc {best_acc:.4f}")



