from data import load_dataloaders
from resnet import Resnet

import torch,torchvision
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR,CosineAnnealingLR,SequentialLR
from torch.optim import AdamW
import torchvision.transforms.v2 as tf

from tqdm import tqdm

print(torchvision.__version__)

torch.set_float32_matmul_precision('high')

 
    # x=torch.randn(3,3,32,32).to("cuda:1")
    # res=Resnet(26,5,46,5,1,2).to("cuda:1")

num_layers=30
proj_kernel=5
residual_channels=30
residual_kernel=5
stride=1
padding=2
bias=True
batch_norm=True
hidden_dim=512
num_classes=10



Device="cuda:3"


resnet_cifar=Resnet(num_layers,
                    proj_kernel,
                    residual_channels,
                    residual_kernel,
                    stride,
                    padding,
                    bias,
                    batch_norm,
                    hidden_dim,
                    num_classes)

trainable_params = sum(p.numel() for p in resnet_cifar.parameters() if p.requires_grad)
print(f"Trainable params: {trainable_params:,}")
# print(1+"sd")

train_loader,test_loader=load_dataloaders(512,num_workers=16)

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


resnet_cifar=resnet_cifar.to(Device)
resnet_cifar = torch.compile(resnet_cifar)
resnet_optim=AdamW(resnet_cifar.parameters(),lr,weight_decay=weight_decay)

warmup=LinearLR(resnet_optim,start_factor=0.01,end_factor=1,total_iters=warmup_epochs)
cosine_decay=CosineAnnealingLR(resnet_optim,num_epochs-warmup_epochs)
scheduler=SequentialLR(resnet_optim,[warmup,cosine_decay],milestones=[warmup_epochs])

criterion=nn.CrossEntropyLoss()

loss_list=[]
prev_loss=torch.tensor(1e16)
best_acc=0.0

for epoch in tqdm(range(num_epochs)):
    count=0
    train_loss=0
    resnet_cifar.train()
    train_correct = 0
    train_total   = 0
    
    for x,y in train_loader:
        x,y=x.to(Device),y.to(Device)
        x=gpu_transforms(x)
        y = y.long()   
        x,y=mixup(x,y)
 
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y_pred=resnet_cifar(x)
            loss=criterion(y_pred, y)

        train_correct+=(y_pred.argmax(1) == y.argmax(1)).sum().item()
        train_total+=y.size(0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(resnet_cifar.parameters(), max_norm=1.0)

        resnet_optim.step()
        resnet_optim.zero_grad()

        train_loss+=loss.item()
        count+=1
    
    scheduler.step()
    resnet_cifar.eval()
    correct = 0
    total   = 0

    train_acc = train_correct / train_total

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(Device), y.to(Device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                y_pred=resnet_cifar(x)
            correct+=(y_pred.argmax(1) == y).sum().item()
            total+=y.size(0)

    acc = correct / total
    print(f"epoch {epoch+1} | loss {train_loss/len(train_loader):.4f} | train_acc {train_acc:.4f} | val_acc {acc:.4f}")


    # ── checkpoint ─────────────────────────────────────
    if acc > best_acc:
        best_acc = acc
        torch.save(resnet_cifar.state_dict(), "best_resnet.pt")
        print(f"  saved checkpoint — best acc {best_acc:.4f}")



