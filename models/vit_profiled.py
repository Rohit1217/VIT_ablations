import torch,torchvision
import torch.nn as nn
import  torch.nn.functional as F


class CNN_block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 group_norm=True):
        super().__init__() 
        
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=bias)
        
        if group_norm:
            self.norm=nn.GroupNorm(4,out_channels)
        else:
            self.norm=nn.Identity()

    def forward(self,x):
        x=F.relu(self.conv(x))
        x=self.norm(x)
        return x


class FFN(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True):
        super().__init__()

        self.fc1=nn.Linear(in_features,out_features,bias)
    
    def forward(self,x):
        x=F.relu(self.x)
        return x



class Self_Attention(nn.Module):
    def __init__(self,embed_dim,n_heads,max_seq_len=1024,causal=False):
        super().__init__()
        self.qkv_proj=nn.Linear(embed_dim,embed_dim*3)
        self.n_heads=n_heads
        self.head_dim=embed_dim//self.n_heads
        self.out_proj=nn.Linear(embed_dim,embed_dim)
        mask=torch.triu(torch.ones(max_seq_len,max_seq_len),diagonal=1).to(dtype=torch.bool)
        self.register_buffer("mask", mask)
        self.causal=causal

            
    def forward(self,x):
        B,C,D=x.shape
        
        #Add Positonal Embedding
        qkv=self.qkv_proj(x)
        qkv=qkv.view(B,C,3,self.n_heads,self.head_dim)
        qkv=qkv.permute(0,3,1,4,2)
        q,k,v=qkv.unbind(-1)

        att_logit=q@torch.transpose(k,2,3) 
        
        if self.causal:
            att_logit=torch.masked_fill(att_logit,self.mask[:C,:C],-torch.inf)
        
        att_logit=att_logit/torch.sqrt(torch.tensor(self.head_dim))
        att=F.softmax(att_logit,dim=-1)
        
        x=att@v
        x=x.permute(0,2,1,3)
        x=x.reshape(B,C,D)
        x=self.out_proj(x)
        return x

class Transformer_block(nn.Module):
    def __init__(self,embed_dim,
                 n_heads,
                 hidden_dim,
                 max_seq_len=1024,
                 causal=False,
                 ):
        super().__init_()

        self.att=Self_Attention(embed_dim,n_heads,max_seq_len,causal)
        self.fc1=nn.Linear(embed_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,embed_dim)
        self.ln1=nn.LayerNorm(embed_dim)
        self.ln2=nn.LayerNorm(embed_dim)

    def forward(self,x):
        x=x+self.att(self.ln1(x))
        residual=x
        
        x=F.gelu(self.fc1(self.ln2(x)))
        x=self.fc2(x)
        x=x+residual
        
        return x

def Position_embeddding(max_seq_len,embed_dim):
    pos_embed=torch.zeros(max_seq_len,embed_dim)
    embed_dim=torch.tensor(embed_dim)
    
    for pos in range(max_seq_len):
        for d in range(embed_dim):
            curr_d=torch.tensor((2*(d+1))//2)
            denom=torch.pow((2*curr_d),embed_dim)

            num=pos
            pos_embed[pos][d]=num/denom
            
    return pos_embed
        
        
print(Position_embeddding(10,3))

# att=Self_Attention(30,5)
# x=torch.randn(2,5,30)
# att=att.to("cuda:0")
# x=x.to("cuda:0")
# x=att(x)
# # mask=torch.triu(torch.ones(3,3),diagonal=1).to(dtype=torch.bool)
# # tensor=torch.randn(3,3)
# # tensor=torch.masked_fill(tensor,mask,1e-64)
# # tensor=F.softmax(tensor,dim=-2)
# print(x.shape)






