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
                 batch_norm=True):
        super().__init__() 
        
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=bias)
        
        if batch_norm:
            self.norm=nn.BatchNorm2d(out_channels)
        else:
            self.norm=nn.Identity()

    def forward(self,x):
        x=F.relu(self.norm(self.conv(x)))
        return x


class FFN(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True):
        super().__init__()

        self.fc1=nn.Linear(in_features,out_features,bias)
    
    def forward(self,x):
        x=F.relu(self.fc1(x))
        return x



class Self_Attention(nn.Module):
    def __init__(self,embed_dim,n_heads,max_seq_len=1024,causal=False):
        super().__init__()
        self.qkv_proj=nn.Linear(embed_dim,embed_dim*3)
        self.n_heads=n_heads
        self.head_dim=embed_dim//self.n_heads
        self.out_proj=nn.Linear(embed_dim,embed_dim)
        self.scale=1/(self.head_dim**0.5)
        
        mask=torch.triu(torch.ones(max_seq_len,max_seq_len),diagonal=1).to(dtype=torch.bool)
        self.register_buffer("mask", mask)
        
        self.causal=causal
        self.att_weights=None

            
    def forward(self,x):
        B,C,D=x.shape
        
        #Add Positonal Embedding
        qkv=self.qkv_proj(x)
        qkv=qkv.view(B,C,3,self.n_heads,self.head_dim)
        qkv=qkv.permute(0,3,1,4,2)
        q,k,v=qkv.unbind(-1)
        # print(q.shape,k.shape)

        att_logit=q@torch.transpose(k,2,3) 
        
        if self.causal:
            att_logit=torch.masked_fill(att_logit,self.mask[:C,:C],-torch.inf)
        
        att_logit*=self.scale
        att=F.softmax(att_logit,dim=-1)
        self.att_weights=att
        
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
        super().__init__()

        self.att=Self_Attention(embed_dim,n_heads,max_seq_len,causal)
        self.fc1=nn.Linear(embed_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,embed_dim)
        self.ln1=nn.LayerNorm(embed_dim)
        self.ln2=nn.LayerNorm(embed_dim)

    def forward(self,x):
        x=x+self.att(self.ln1(x))
        residual=x
        
        x=F.gelu(self.fc1(self.ln2(x)))
        x=F.dropout(x, p=0.1, training=self.training) 
        x=self.fc2(x)
        x=x+residual
        
        return x


def Position_embedding(max_seq_len,embed_dim):
    pos_embed=torch.zeros(max_seq_len,embed_dim)
    embed_dim=torch.tensor(embed_dim)
    k=10000
    
    for pos in range(max_seq_len):
        for d in range(embed_dim):
            num=torch.tensor(pos)
            
            if d%2==0:
                curr_exp=(d)/embed_dim
                denom=torch.pow(k,curr_exp)
                pos_embed[pos][d]=torch.sin(num/denom)
            else:
                curr_exp=((d-1))/embed_dim
                denom=torch.pow(k,curr_exp)
                pos_embed[pos][d]=torch.cos(num/denom)

    return pos_embed


class Transformer(nn.Module):
    def __init__(self,num_blocks,
                 embed_dim,
                 n_heads,
                 hidden_dim,
                 max_seq_len=1024,
                 causal=False,
                 ):
        super().__init__()

        pos_embed=Position_embedding(max_seq_len,embed_dim)
        self.register_buffer("pos_embed", pos_embed)
        self.transformer_block_list=nn.ModuleList([Transformer_block(embed_dim,
                                                                n_heads,
                                                                hidden_dim,
                                                                max_seq_len,
                                                                causal) for n in range(num_blocks)])
        
    def forward(self,x):
        B,C,D=x.shape
        x=x+self.pos_embed[:C]
        
        for transformer_block in self.transformer_block_list:
            x=transformer_block(x)
        
        return x



if __name__=="__main__":
    transformer=Transformer(4,30,5,120).to("cuda:1")
    x=torch.randn(2,5,30).to("cuda:1")
    print(transformer(x).shape)
        







