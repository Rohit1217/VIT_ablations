import torch,torchvision
import torch.nn as nn
import  torch.nn.functional as F
from models import Transformer_block,Position_embedding


class Patchify(nn.Module):  #ONLY WORKS FOR SQUARE IMAGES
    def __init__(self, 
                 image_size,
                 patch_size,
                 embed_dim,
                 patch_overlap=False):
        super().__init__()

        self.image_size=image_size
        self.patch_size=patch_size
        self.embed_dim=embed_dim
        # self.num_patches=(self.image_size//self.patch_size)**2
        
        if patch_overlap:
            self.conv_proj=nn.Conv2d(3,self.embed_dim,self.patch_size,self.patch_size//2)
        else:
            self.conv_proj=nn.Conv2d(3,self.embed_dim,self.patch_size,self.patch_size)
    
    def forward(self,x):
        x=self.conv_proj(x)
        x=x.flatten(2)
        x=x.transpose(1,2)
        return x


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

        cls_token=self.cls_token.expand(x.shape[0],-1,-1)
        if self.cls_pos=="prepend":
            x=torch.concat([cls_token,x],dim=1)
        else:
            x=torch.concat([x,cls_token],dim=1)

        if self.pos_embed_type=="Sin":
            x+=self.pos_embed[None,:C+1,:]
        else:
            x+=self.pos_embed

        for transformer_block in self.transformer_block_list:
            x=transformer_block(x)
        if not self.use_cls:
            x_cls=self.cls(self.ln_final(x[:, 1:-1, :].mean(dim=1) if self.cls_pos=="prepend" else x[:, :-1, :].mean(dim=1)))
        else:
            x_cls=self.cls(self.ln_final(x[:, 0, :] if self.cls_pos=="prepend" else x[:, -1, :]))
        return x_cls
        
    
        

if __name__=="__main__":
    patchifier=Patchify(32,4,10)
    random_im=torch.randn(5,3,32,32).to("cuda:0")
    vit=VIT(32,4,10,6,32,4,32*4).to("cuda:0")
    cls=vit(random_im)
    print(cls,cls.shape)




