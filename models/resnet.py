import torchvision,torch
import torch.nn as nn
from models import CNN_block,FFN
import torch.nn.functional as F


class Resnet_block(nn.Module):
    def __init__(self,
        in_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        batch_norm=True):
        super().__init__()

        self.bn1=nn.BatchNorm2d(in_channels)
        self.cnn_block1=CNN_block(in_channels,in_channels,kernel_size,stride,padding,bias,batch_norm)
        self.cnn_block2=CNN_block(in_channels,in_channels,kernel_size,stride,padding,bias,batch_norm)
        self.cnn_out=nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,bias=True)

    def forward(self,x):
        residual=x
        x=self.bn1(x)
        x=self.cnn_block1(x)
        x=self.cnn_block2(x)
        x=self.cnn_out(x)
        x=x+residual
        return x

class Resnet_big_block(nn.Module):
    def __init__(self,
                    num_layers,
                    residual_channels,
                    residual_kernel,
                    stride=1,
                    padding=0,
                    bias=True,
                    batch_norm=True):
        super().__init__()
        
        self.res_list=nn.ModuleList([Resnet_block(residual_channels,
                                            residual_kernel,
                                            stride,padding,
                                            bias,
                                            batch_norm) for idx in range(num_layers)])
    
        self.conv_proj=nn.Conv2d(residual_channels,residual_channels*2,3,2,1)

    def forward(self,x):
        for res in self.res_list:
            x=res(x)
        x=self.conv_proj(x)
        return x

# class Resnet(nn.Module):
#     def __init__(self,
#                  num_layers,
#                  proj_kernel,
#                  residual_channels,
#                  residual_kernel,
#                  stride=1,
#                  padding=0,
#                  bias=True,
#                  batch_norm=True,
#                  hidden_dim=512,
#                  num_classes=10):
#         super().__init__()
#         self.conv_proj=nn.Conv2d(3,residual_channels,proj_kernel)
        
#         self.res_list=nn.ModuleList([Resnet_block(residual_channels,
#                                                   residual_kernel,
#                                                     stride,padding,
#                                                     bias,
#                                                     batch_norm) for idx in range(num_layers)])

#         self.bn_proj_out=nn.BatchNorm2d(residual_channels)
#         self.conv_proj_out=nn.Conv2d(residual_channels,10,residual_kernel,2)
        
#         self.flatten=nn.Flatten()
#         self.bn_out=nn.BatchNorm1d(1440)
        
#         self.fc1=FFN(1440,hidden_dim)
#         self.fc2=nn.Linear(hidden_dim,num_classes)


#     def forward(self,x):
#         x=self.conv_proj(x)
    
#         for res in self.res_list:
#             x=res(x)    

#         x=self.bn_proj_out(x)
#         x=F.relu(self.conv_proj_out(x))
#         x=self.bn_out(self.flatten(x))
#         x=self.fc2(self.fc1(x))

#         return x

class Resnet(nn.Module):
    def __init__(self,
                 num_layers,
                 proj_kernel,
                 residual_channels,
                 residual_kernel,
                 stride=1,
                 padding=0,
                 bias=True,
                 batch_norm=True,
                 hidden_dim=512,
                 num_classes=10):
        super().__init__()
        self.conv_proj=nn.Conv2d(3,residual_channels//2,proj_kernel)
        
        
        self.res1=Resnet_big_block(num_layers//3,residual_channels//2,residual_kernel,stride,padding,bias,batch_norm)
        self.res2=Resnet_big_block(num_layers//3,residual_channels,residual_kernel,stride,padding,bias,batch_norm)
        self.res3=Resnet_big_block(num_layers//3,residual_channels*2,residual_kernel,stride,padding,bias,batch_norm)
        
        self.flatten=nn.Flatten()
        self.bn_out=nn.BatchNorm1d(1920)
        
        self.fc1=FFN(1920,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,num_classes)


    def forward(self,x):
        x=self.conv_proj(x)
    
        x=self.res3(self.res2(self.res1(x)))

        x=self.bn_out(self.flatten(x))
        x=self.fc2(self.fc1(x))

        return x




if __name__=="__main__":
    x=torch.randn(3,3,32,32).to("cuda:3")
    res=Resnet(30,5,30,5,1,2).to("cuda:3")
    # print(res)
    print(sum(p.numel() for p in res.parameters() if p.requires_grad))
    print(res(x))
